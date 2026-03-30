import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';
import 'package:intl/intl.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'firebase_options.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:fl_chart/fl_chart.dart';

// ─────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────
const String kBroker    = 'broker.emqx.io';
const String kTopic = 'tinyml/quang_wm_2026/status';
const int    kPort      = 1883;
const int    kHeartbeatTimeout = 60; // seconds
const int    kAlarmDebounce    = 6;
const int    kMaxMaeHistory    = 40;
const int    kMaxHistory       = 30;

// MAE thresholds per mode (phải khớp firmware)
const Map<String, double> kThresholds = {
  'GENTLE': 0.051,
  'STRONG': 0.095,
  'SPIN':   0.083,
};

// ─────────────────────────────────────────────
// MODELS
// ─────────────────────────────────────────────
enum MachineStatus { waiting, ok, alarm, deviceLost }
enum WashMode { gentle, strong, spin, unknown }

extension WashModeX on WashMode {
  String get label {
    switch (this) {
      case WashMode.gentle: return 'GENTLE';
      case WashMode.strong: return 'STRONG';
      case WashMode.spin:   return 'SPIN';
      default:              return '—';
    }
  }
  String get description {
    switch (this) {
      case WashMode.gentle: return 'Giặt nhẹ / thấm';
      case WashMode.strong: return 'Giặt chính';
      case WashMode.spin:   return 'Vắt cao tốc';
      default:              return '';
    }
  }
  Color get color {
    switch (this) {
      case WashMode.gentle: return const Color(0xFF22D9A0);
      case WashMode.strong: return const Color(0xFF4A9EFF);
      case WashMode.spin:   return const Color(0xFFF5A623);
      default:              return const Color(0xFF6B7285);
    }
  }
}

WashMode parseModeFromString(String? s) {
  switch (s?.toUpperCase()) {
    case 'GENTLE': return WashMode.gentle;
    case 'STRONG': return WashMode.strong;
    case 'SPIN':   return WashMode.spin;
    default:       return WashMode.unknown;
  }
}

class AlarmEvent {
  final bool isAlarm;
  final double mae;
  final int win;
  final int consec;
  final WashMode mode;
  final DateTime time;
  bool acknowledged;

  AlarmEvent({
    required this.isAlarm,
    required this.mae,
    required this.win,
    required this.consec,
    required this.mode,
    required this.time,
    this.acknowledged = false,
  });
}

// ─────────────────────────────────────────────
// NOTIFICATIONS
// ─────────────────────────────────────────────
final FlutterLocalNotificationsPlugin _notifPlugin =
    FlutterLocalNotificationsPlugin();

Future<void> initNotifications() async {
  const android = AndroidInitializationSettings('@mipmap/ic_launcher');
  await _notifPlugin.initialize(
    const InitializationSettings(android: android),
  );
  await _notifPlugin
      .resolvePlatformSpecificImplementation<
          AndroidFlutterLocalNotificationsPlugin>()
      ?.requestNotificationsPermission();
}

Future<void> showAlarmNotification(double mae) async {
  const details = AndroidNotificationDetails(
    'alarm_channel', 'Cảnh báo rung lắc',
    importance: Importance.max,
    priority: Priority.high,
    color: Color(0xFFFF4C6A),
    playSound: true,
    enableVibration: true,
  );
  await _notifPlugin.show(
    0,
    '⚠️ MÁY GIẶT BẤT THƯỜNG!',
    'Phát hiện rung lắc mạnh (MAE: ${mae.toStringAsFixed(4)}). Kiểm tra ngay!',
    const NotificationDetails(android: details),
  );
}

// ─────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────
void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  try {
    await Firebase.initializeApp(
        options: DefaultFirebaseOptions.currentPlatform);
  } catch (e) {
    debugPrint('❌ Firebase init: $e');
  }

  await initNotifications();
  runApp(const TinyMLApp());
}

// ─────────────────────────────────────────────
// APP ROOT
// ─────────────────────────────────────────────
class TinyMLApp extends StatelessWidget {
  const TinyMLApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Edge AI Monitor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0A0C12),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF0A0C12),
          elevation: 0,
          scrolledUnderElevation: 0,
        ),
        colorScheme: const ColorScheme.dark(
          surface: Color(0xFF111520),
          primary: Color(0xFF22D9A0),
          error: Color(0xFFFF4C6A),
        ),
      ),
      home: const MonitorScreen(),
    );
  }
}

// ─────────────────────────────────────────────
// MONITOR SCREEN
// ─────────────────────────────────────────────
class MonitorScreen extends StatefulWidget {
  const MonitorScreen({super.key});

  @override
  State<MonitorScreen> createState() => _MonitorScreenState();
}

class _MonitorScreenState extends State<MonitorScreen>
    with TickerProviderStateMixin {
  // MQTT
  MqttServerClient? _client;

  // State
  MachineStatus _status         = MachineStatus.waiting;
  WashMode      _mode           = WashMode.unknown;
  bool          _isConnected    = false;
  double        _currentMae     = 0.0;
  int           _currentWin     = 0;
  int           _currentConsec  = 0;

  // Stats
  int    _totalWins   = 0;
  int    _alarmCount  = 0;
  DateTime? _connectedAt;

  // History
  final List<AlarmEvent>  _events     = [];
  final List<double>      _maeHistory = [];
  final List<bool>        _uptimeSegs = []; // true=ok, false=alarm

  // Phase counters
  final Map<WashMode, int> _phaseCount = {
    WashMode.gentle: 0,
    WashMode.strong: 0,
    WashMode.spin:   0,
  };

  // Heartbeat
  Timer? _heartbeatTimer;
  Timer? _clockTimer;

  // Tab
  int _tabIndex = 0;

  // Alarm animation
  late AnimationController _alarmAnim;
  late Animation<double>   _alarmGlow;

  @override
  @override
  void initState() {
    super.initState();

    _alarmAnim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat(reverse: true);

    _alarmGlow = Tween<double>(begin: 0.1, end: 0.4).animate(
      CurvedAnimation(parent: _alarmAnim, curve: Curves.easeInOut),
    );

    // --- THÊM ĐOẠN NÀY VÀO ---
    // Cứ 1 giây sẽ ép giao diện cập nhật 1 lần
    _clockTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      // Tối ưu hóa: Chỉ gọi setState khi app đang mở và người dùng đang ở Tab Thống kê (_tabIndex == 2)
      if (mounted && _tabIndex == 2 && _isConnected) {
        setState(() {}); 
      }
    });
    // -------------------------

    _connectMQTT();
  }

  @override
  void dispose() {
    _client?.disconnect();
    _heartbeatTimer?.cancel();
    _clockTimer?.cancel();
    _alarmAnim.dispose();
    super.dispose();
  }

  // ── HEARTBEAT WATCHDOG ──────────────────────
  void _resetHeartbeat() {
    _heartbeatTimer?.cancel();
    _heartbeatTimer = Timer(
      const Duration(seconds: kHeartbeatTimeout),
      () {
        if (!mounted) return;
        setState(() => _status = MachineStatus.deviceLost);
      },
    );
  }

  // ── MQTT ────────────────────────────────────
  Future<void> _connectMQTT() async {
    _client = MqttServerClient(
      kBroker,
      'android_${DateTime.now().millisecondsSinceEpoch}',
    );
    _client!.port            = kPort;
    _client!.keepAlivePeriod = 20;
    _client!.autoReconnect   = true;

    _client!.onAutoReconnect = () =>
        debugPrint('[MQTT] Auto-reconnecting...');

    _client!.onAutoReconnected = () {
      if (!mounted) return;
      setState(() {
        _isConnected = true;
        if (_status != MachineStatus.alarm) {
          _status = MachineStatus.ok;
        }
      });
    };

    _client!.onDisconnected = () {
      if (!mounted) return;
      setState(() {
        _isConnected   = false;
        _status        = MachineStatus.waiting;
      });
    };

    try {
      await _client!.connect();
    } catch (e) {
      _client!.disconnect();
      return;
    }

    if (_client!.connectionStatus!.state == MqttConnectionState.connected) {
      setState(() {
        _isConnected  = true;
        _status       = MachineStatus.ok;
        _connectedAt  = DateTime.now();
      });

      _client!.subscribe(kTopic, MqttQos.atMostOnce);
      _resetHeartbeat();

      _client!.updates!
          .listen((List<MqttReceivedMessage<MqttMessage?>>? c) {
        final raw = c![0].payload as MqttPublishMessage;
        final payload = MqttPublishPayload.bytesToStringAsString(
            raw.payload.message);
        _parseData(payload);
      });
    }
  }

  // ── PARSE MQTT PAYLOAD ──────────────────────
  void _parseData(String payload) {
    debugPrint('👉 RAW MQTT NHẬN ĐƯỢC: $payload');
    try {
      final data    = jsonDecode(payload) as Map<String, dynamic>;
      final isAlarm = data['is_alarm'] as bool? ?? false;
      final mae     = (data['mae'] ?? 0.0).toDouble();
      final win     = data['win'] as int? ?? 0;
      final consec  = data['consec'] as int? ?? 0;
      final mode    = parseModeFromString(data['mode'] as String?);

      _resetHeartbeat();

      setState(() {
        _totalWins++;
        _currentWin    = win;
        _currentConsec = consec;
        _currentMae    = mae;

        // MAE history
        _maeHistory.add(mae);
        if (_maeHistory.length > kMaxMaeHistory) _maeHistory.removeAt(0);

        // Mode
        if (mode != WashMode.unknown) {
          _mode = mode;
          _phaseCount[mode] = (_phaseCount[mode] ?? 0) + 1;
        }

        if (isAlarm) {
          // Only log edge: waiting/ok → alarm
          if (_status != MachineStatus.alarm) {
            _addEvent(AlarmEvent(
              isAlarm: true, mae: mae, win: win,
              consec: consec, mode: _mode, time: DateTime.now(),
            ));
            _alarmCount++;
            _saveToFirestore(mae, win, consec, isAlarm: true);
            showAlarmNotification(mae);
          }
          _status = MachineStatus.alarm;
          _uptimeSegs.add(false);
        } else {
          // Edge: alarm → ok
          if (_status == MachineStatus.alarm) {
            _addEvent(AlarmEvent(
              isAlarm: false, mae: mae, win: win,
              consec: consec, mode: _mode, time: DateTime.now(),
            ));
            _saveToFirestore(mae, win, consec, isAlarm: false);
          }
          _status        = MachineStatus.ok;
          _currentMae    = mae;
          _currentWin    = win;
          _currentConsec = consec;
          _uptimeSegs.add(true);
        }

        if (_uptimeSegs.length > 100) _uptimeSegs.removeAt(0);
      });
    } catch (e) {
      debugPrint('❌ JSON parse: $e');
    }
  }

  void _addEvent(AlarmEvent e) {
    _events.insert(0, e);
    if (_events.length > kMaxHistory) _events.removeLast();
  }

  // ── FIRESTORE ───────────────────────────────
  Future<void> _saveToFirestore(
    double mae, int win, int consec, {required bool isAlarm}
  ) async {
    try {
      await FirebaseFirestore.instance.collection('alarm_history').add({
        'timestamp': FieldValue.serverTimestamp(),
        'is_alarm':  isAlarm,
        'mae':       mae,
        'win':       win,
        'consec':    consec,
        'mode':      _mode.label,
      });
    } catch (e) {
      debugPrint('❌ Firestore: $e');
    }
  }

  // ─────────────────────────────────────────────
  // BUILD
  // ─────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0C12),
      body: SafeArea(
        child: Column(children: [
          _buildHeader(),
          _buildTabBar(),
          Expanded(child: _buildBody()),
        ]),
      ),
    );
  }

  // ── HEADER ──────────────────────────────────
  Widget _buildHeader() {
    final (color, text) = switch (_status) {
      MachineStatus.deviceLost => (const Color(0xFFFF4C6A), 'DEVICE LOST'),
      _ when _isConnected      => (const Color(0xFF22D9A0), 'ONLINE'),
      _                        => (const Color(0xFF6B7285), 'OFFLINE'),
    };

    return Container(
      padding: const EdgeInsets.fromLTRB(20, 14, 16, 14),
      decoration: const BoxDecoration(
        border: Border(bottom: BorderSide(color: Color(0x12FFFFFF))),
      ),
      child: Row(children: [
        Container(
          width: 32, height: 32,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(8),
            gradient: const LinearGradient(
              colors: [Color(0xFF22D9A0), Color(0xFF4A9EFF)],
              begin: Alignment.topLeft, end: Alignment.bottomRight,
            ),
          ),
          child: const Center(child: Text('⚙️', style: TextStyle(fontSize: 16))),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text('Edge AI Monitor',
                style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700,
                    letterSpacing: -0.3, color: Color(0xFFE8EAF0))),
            Text(kBroker,
                style: const TextStyle(fontSize: 11, color: Color(0xFF6B7285),
                    fontFamily: 'monospace')),
          ]),
        ),
        _ConnBadge(color: color, text: text, isOnline: _isConnected),
      ]),
    );
  }

  // ── TAB BAR ─────────────────────────────────
  Widget _buildTabBar() {
    const tabs = ['Monitor', 'MAE Chart', 'Thống kê'];
    return Container(
      decoration: const BoxDecoration(
        border: Border(bottom: BorderSide(color: Color(0x12FFFFFF))),
      ),
      child: Row(
        children: List.generate(tabs.length, (i) {
          final active = i == _tabIndex;
          return GestureDetector(
            onTap: () => setState(() => _tabIndex = i),
            child: Container(
              padding: const EdgeInsets.fromLTRB(16, 10, 16, 12),
              decoration: BoxDecoration(
                border: Border(
                  bottom: BorderSide(
                    color: active ? const Color(0xFF22D9A0) : Colors.transparent,
                    width: 2,
                  ),
                ),
              ),
              child: Text(tabs[i],
                  style: TextStyle(
                    fontSize: 12, fontWeight: FontWeight.w700,
                    fontFamily: 'monospace', letterSpacing: 0.8,
                    color: active ? const Color(0xFFE8EAF0) : const Color(0xFF6B7285),
                  )),
            ),
          );
        }),
      ),
    );
  }

  // ── BODY ────────────────────────────────────
  Widget _buildBody() {
    return switch (_tabIndex) {
      1     => _ChartTab(maeHistory: _maeHistory, mode: _mode),
      2     => _StatsTab(
                totalWins: _totalWins, alarmCount: _alarmCount,
                connectedAt: _connectedAt, phaseCount: _phaseCount,
                uptimeSegs: _uptimeSegs, maeHistory: _maeHistory,
               ),
      _     => _buildMonitorTab(),
    };
  }

  // ── MONITOR TAB ─────────────────────────────
  Widget _buildMonitorTab() {
    return ListView(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
      children: [
        _buildStatusRing(),
        const SizedBox(height: 8),
        _buildModeBadge(),
        const SizedBox(height: 20),

        // Heartbeat warning
        if (_status == MachineStatus.deviceLost) ...[
          _InfoBanner(
            color: const Color(0xFFF5A623),
            icon: '⚠️',
            message: 'Không nhận được tín hiệu >60s — kiểm tra thiết bị',
          ),
          const SizedBox(height: 12),
        ],

        // Alarm detail card
        if (_status == MachineStatus.alarm) ...[
          _buildAlarmCard(),
          const SizedBox(height: 16),
        ],

        // Phase row
        _SectionHeader(title: 'Pha giặt'),
        const SizedBox(height: 10),
        _buildPhaseRow(),
        const SizedBox(height: 20),

        // History
        _SectionHeader(title: 'Lịch sử sự kiện'),
        const SizedBox(height: 10),
        _buildHistory(),
      ],
    );
  }

  // ── STATUS RING ─────────────────────────────
  Widget _buildStatusRing() {
    final (color, icon, label) = switch (_status) {
      MachineStatus.ok       => (const Color(0xFF22D9A0), '✓', 'OK'),
      MachineStatus.alarm    => (const Color(0xFFFF4C6A), '⚡', 'ALARM'),
      MachineStatus.deviceLost => (const Color(0xFFF5A623), '💀', 'LOST'),
      _                      => (const Color(0xFF6B7285), '⏳', 'WAITING'),
    };

    return Center(
      child: AnimatedBuilder(
        animation: _alarmAnim,
        builder: (_, __) {
          final glow = _status == MachineStatus.alarm
              ? _alarmGlow.value : 0.1;
          return Container(
            width: 200, height: 200,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: color.withOpacity(glow * 0.5),
              border: Border.all(color: color.withOpacity(0.4), width: 3),
              boxShadow: [
                BoxShadow(
                  color: color.withOpacity(glow),
                  blurRadius: 40, spreadRadius: 4,
                ),
              ],
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(icon, style: const TextStyle(fontSize: 48)),
                const SizedBox(height: 6),
                Text(label,
                    style: TextStyle(
                      fontSize: 22, fontWeight: FontWeight.w900,
                      color: color, letterSpacing: 3,
                      fontFamily: 'monospace',
                    )),
                if (_status == MachineStatus.alarm)
                  Padding(
                    padding: const EdgeInsets.only(top: 4),
                    child: Text('MAE ${_currentMae.toStringAsFixed(4)}',
                        style: TextStyle(
                          fontSize: 12, color: color.withOpacity(0.8),
                          fontFamily: 'monospace',
                        )),
                  ),
              ],
            ),
          );
        },
      ),
    );
  }

  // ── MODE BADGE ──────────────────────────────
  Widget _buildModeBadge() {
    return Center(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(20),
          color: const Color(0xFF181D2A),
          border: Border.all(color: const Color(0x20FFFFFF)),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Container(
            width: 7, height: 7,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: _mode == WashMode.unknown
                  ? const Color(0xFF6B7285) : _mode.color,
            ),
          ),
          const SizedBox(width: 8),
          Text(
            _mode == WashMode.unknown
                ? 'Chờ kết nối...'
                : '${_mode.label} — ${_mode.description}',
            style: TextStyle(
              fontSize: 12, fontFamily: 'monospace', fontWeight: FontWeight.w700,
              color: _mode == WashMode.unknown
                  ? const Color(0xFF6B7285) : _mode.color,
            ),
          ),
        ]),
      ),
    );
  }

  // ── ALARM CARD ──────────────────────────────
  Widget _buildAlarmCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0x14FF4C6A),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0x40FF4C6A)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          const Text('⚡ CHI TIẾT BẤT THƯỜNG',
              style: TextStyle(
                fontSize: 10, fontWeight: FontWeight.w700,
                letterSpacing: 1.5, fontFamily: 'monospace',
                color: Color(0xFFFF4C6A),
              )),
          const Spacer(),
          GestureDetector(
            onTap: _acknowledgeAlarm,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: const Color(0x20FF4C6A),
                borderRadius: BorderRadius.circular(6),
                border: Border.all(color: const Color(0x40FF4C6A)),
              ),
              child: const Text('✓ XÁC NHẬN',
                  style: TextStyle(fontSize: 10, fontFamily: 'monospace',
                      fontWeight: FontWeight.w700, color: Color(0xFFFF4C6A))),
            ),
          ),
        ]),
        const SizedBox(height: 12),
        Row(children: [
          _MetricCell(label: 'MAE', value: _currentMae.toStringAsFixed(4),
              color: const Color(0xFFFF4C6A)),
          const SizedBox(width: 8),
          _MetricCell(label: 'WINDOW', value: '#$_currentWin',
              color: const Color(0xFFF5A623)),
          const SizedBox(width: 8),
          _MetricCell(label: 'CONSEC', value: '$_currentConsec wins',
              color: const Color(0xFFE8EAF0)),
        ]),
      ]),
    );
  }

  void _acknowledgeAlarm() {
    final idx = _events.indexWhere(
        (e) => e.isAlarm && e.win == _currentWin);
    if (idx >= 0) {
      setState(() => _events[idx].acknowledged = true);
    }
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: const Text('✓ Đã xác nhận — đánh dấu để cải thiện model',
            style: TextStyle(fontFamily: 'monospace', fontSize: 13)),
        backgroundColor: const Color(0xFF181D2A),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        duration: const Duration(seconds: 3),
      ),
    );
  }

  // ── PHASE ROW ───────────────────────────────
  Widget _buildPhaseRow() {
    return Row(children: [
      _PhaseCell(mode: WashMode.gentle, isActive: _mode == WashMode.gentle),
      const SizedBox(width: 8),
      _PhaseCell(mode: WashMode.strong, isActive: _mode == WashMode.strong),
      const SizedBox(width: 8),
      _PhaseCell(mode: WashMode.spin, isActive: _mode == WashMode.spin),
    ]);
  }

  // ── HISTORY ─────────────────────────────────
  Widget _buildHistory() {
    if (_events.isEmpty) {
      return Container(
        padding: const EdgeInsets.symmetric(vertical: 32),
        alignment: Alignment.center,
        child: const Text('Hệ thống hoạt động ổn định...',
            style: TextStyle(fontSize: 13, color: Color(0xFF6B7285),
                fontFamily: 'monospace')),
      );
    }

    return Column(
      children: _events.map((e) => _buildHistoryItem(e)).toList(),
    );
  }

  Widget _buildHistoryItem(AlarmEvent e) {
    final color = e.isAlarm ? const Color(0xFFFF4C6A) : const Color(0xFF22D9A0);
    final icon  = e.isAlarm ? '⚡' : '✓';
    final title = e.isAlarm
        ? 'Phát hiện rung lắc bất thường${e.acknowledged ? ' [xác nhận]' : ''}'
        : 'Hệ thống phục hồi bình thường';
    final timeStr = DateFormat('HH:mm:ss — dd/MM').format(e.time);

    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      decoration: BoxDecoration(
        color: const Color(0xFF181D2A),
        borderRadius: BorderRadius.circular(12),
        border: Border(left: BorderSide(color: color, width: 3)),
      ),
      child: ListTile(
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 14, vertical: 4),
        leading: Container(
          width: 38, height: 38,
          decoration: BoxDecoration(
            color: color.withOpacity(0.12),
            borderRadius: BorderRadius.circular(10),
          ),
          child: Center(child: Text(icon)),
        ),
        title: Text(title,
            style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600,
                color: Color(0xFFE8EAF0))),
        subtitle: Text('$timeStr · ${e.mode.label}',
            style: const TextStyle(fontSize: 11, color: Color(0xFF6B7285),
                fontFamily: 'monospace')),
        trailing: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(e.mae.toStringAsFixed(4),
                style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700,
                    fontFamily: 'monospace', color: color)),
            Text('Win #${e.win}',
                style: const TextStyle(fontSize: 11, color: Color(0xFF6B7285),
                    fontFamily: 'monospace')),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────
// CHART TAB
// ─────────────────────────────────────────────
class _ChartTab extends StatelessWidget {
  final List<double> maeHistory;
  final WashMode mode;

  const _ChartTab({required this.maeHistory, required this.mode});

  @override
  Widget build(BuildContext context) {
    final spots = maeHistory.asMap().entries
        .map((e) => FlSpot(e.key.toDouble(), e.value))
        .toList();

    final thr = kThresholds[mode.label] ?? 0.15;

    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        _SectionHeader(title: 'MAE theo thời gian (${maeHistory.length} điểm)'),
        const SizedBox(height: 12),

        // Legend
        Wrap(
          spacing: 16,
          runSpacing: 10,
          children: [
            _LegendDot(color: WashMode.gentle.color, label: 'THR Gentle ${kThresholds['GENTLE']}'),
            _LegendDot(color: WashMode.strong.color, label: 'THR Strong ${kThresholds['STRONG']}'),
            _LegendDot(color: WashMode.spin.color, label: 'THR Spin ${kThresholds['SPIN']}'),
          ],
        ),
        const SizedBox(height: 14),

        Container(
          height: 200,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: const Color(0xFF111520),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: const Color(0x12FFFFFF)),
          ),
          child: spots.isEmpty
              ? const Center(
                  child: Text('Chưa có dữ liệu...',
                      style: TextStyle(color: Color(0xFF6B7285),
                          fontFamily: 'monospace', fontSize: 13)))
              : LineChart(LineChartData(
                  minY: 0, maxY: 0.5,
                  gridData: FlGridData(
                    show: true,
                    getDrawingHorizontalLine: (_) =>
                        const FlLine(color: Color(0x0CFFFFFF), strokeWidth: 1),
                    getDrawingVerticalLine: (_) =>
                        const FlLine(color: Color(0x0CFFFFFF), strokeWidth: 1),
                  ),
                  borderData: FlBorderData(show: false),
                  titlesData: FlTitlesData(
                    topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                    rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                    bottomTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                    leftTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true, reservedSize: 36,
                        getTitlesWidget: (v, _) => Text(
                          v.toStringAsFixed(1),
                          style: const TextStyle(fontSize: 10,
                              color: Color(0xFF6B7285), fontFamily: 'monospace'),
                        ),
                      ),
                    ),
                  ),
                  extraLinesData: ExtraLinesData(horizontalLines: [
                    HorizontalLine(y: kThresholds['GENTLE']!, color: WashMode.gentle.color,
                        strokeWidth: 1, dashArray: [4, 4]),
                    HorizontalLine(y: kThresholds['STRONG']!, color: WashMode.strong.color,
                        strokeWidth: 1, dashArray: [4, 4]),
                    HorizontalLine(y: kThresholds['SPIN']!, color: WashMode.spin.color,
                        strokeWidth: 1, dashArray: [4, 4]),
                  ]),
                  lineBarsData: [
                    LineChartBarData(
                      spots: spots,
                      isCurved: true, curveSmoothness: 0.3,
                      color: const Color(0xFFE8EAF0), // Đổi sang trắng xám
                      barWidth: 2,
                      belowBarData: BarAreaData(
                        show: true,
                        color: const Color(0x14FFFFFF), // Sáng mờ nhẹ ở dưới
                      ),
                      dotData: FlDotData(
                        show: true,
                        getDotPainter: (spot, _, __, ___) =>
                            FlDotCirclePainter(
                          radius: 3,
                          color: spot.y > thr
                              ? const Color(0xFFFF4C6A)
                              : const Color(0xFF4A9EFF),
                          strokeWidth: 0,
                        ),
                      ),
                    ),
                  ],
                )),
        ),

        const SizedBox(height: 24),
        _SectionHeader(title: 'Phân phối MAE theo pha'),
        const SizedBox(height: 12),
        // (Additional phase bar chart can be added here if fl_chart bar is available)
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: const Color(0xFF111520),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: const Color(0x12FFFFFF)),
          ),
          child: const Text('Xem tab Thống kê để biết phân bổ theo pha.',
              style: TextStyle(fontSize: 13, color: Color(0xFF6B7285),
                  fontFamily: 'monospace')),
        ),
      ],
    );
  }
}

// ─────────────────────────────────────────────
// STATS TAB
// ─────────────────────────────────────────────
class _StatsTab extends StatelessWidget {
  final int totalWins;
  final int alarmCount;
  final DateTime? connectedAt;
  final Map<WashMode, int> phaseCount;
  final List<bool> uptimeSegs;
  final List<double> maeHistory;

  const _StatsTab({
    required this.totalWins, required this.alarmCount,
    required this.connectedAt, required this.phaseCount,
    required this.uptimeSegs, required this.maeHistory,
  });

  String get _uptime {
    if (connectedAt == null) return '—';
    final d = DateTime.now().difference(connectedAt!);
    final m = d.inMinutes, s = d.inSeconds % 60;
    return '${m}m ${s}s';
  }

  double get _avgMae {
    if (maeHistory.isEmpty) return 0;
    return maeHistory.reduce((a, b) => a + b) / maeHistory.length;
  }

  String get _alarmRate {
    if (totalWins == 0) return '0.0%';
    return '${(alarmCount / totalWins * 100).toStringAsFixed(1)}%';
  }

  @override
  Widget build(BuildContext context) {
    final segs = uptimeSegs.length > 24
        ? uptimeSegs.sublist(uptimeSegs.length - 24)
        : uptimeSegs;

    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        // Stat cards grid
        GridView.count(
          crossAxisCount: 2,
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          crossAxisSpacing: 10, mainAxisSpacing: 10,
          childAspectRatio: 1.6,
          children: [
            _StatCard(label: 'Tổng windows', value: '$totalWins',
                sub: 'inference cycles', color: const Color(0xFF4A9EFF)),
            _StatCard(label: 'Alarm events', value: '$alarmCount',
                sub: 'tỉ lệ: $_alarmRate', color: const Color(0xFFFF4C6A)),
            _StatCard(label: 'MAE trung bình',
                value: maeHistory.isEmpty ? '—' : _avgMae.toStringAsFixed(4),
                sub: 'tất cả pha', color: const Color(0xFFF5A623)),
            _StatCard(label: 'Uptime', value: _uptime,
                sub: connectedAt != null
                    ? 'kể từ ${DateFormat('HH:mm').format(connectedAt!)}'
                    : 'chưa kết nối',
                color: const Color(0xFF22D9A0)),
          ],
        ),

        const SizedBox(height: 20),

        // Uptime bar
        _SectionHeader(title: 'Trạng thái theo thời gian (24 phân đoạn)'),
        const SizedBox(height: 10),
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: const Color(0xFF111520),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: const Color(0x12FFFFFF)),
          ),
          child: Row(
            children: segs.isEmpty
                ? [const Expanded(child: SizedBox(height: 8))]
                : segs.map((ok) => Expanded(
                      child: Container(
                        height: 8, margin: const EdgeInsets.symmetric(horizontal: 1),
                        decoration: BoxDecoration(
                          color: ok
                              ? const Color(0xFF22D9A0)
                              : const Color(0xFFFF4C6A),
                          borderRadius: BorderRadius.circular(2),
                        ),
                      ),
                    )).toList(),
          ),
        ),

        const SizedBox(height: 20),

        // Phase breakdown
        _SectionHeader(title: 'Phân bổ theo pha'),
        const SizedBox(height: 10),
        ...WashMode.values
            .where((m) => m != WashMode.unknown)
            .map((m) => _PhaseBar(
                  mode: m,
                  count: phaseCount[m] ?? 0,
                  total: totalWins,
                )),
      ],
    );
  }
}

// ─────────────────────────────────────────────
// SMALL WIDGETS
// ─────────────────────────────────────────────
class _ConnBadge extends StatelessWidget {
  final Color color;
  final String text;
  final bool isOnline;
  const _ConnBadge({required this.color, required this.text, required this.isOnline});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Container(
          width: 6, height: 6,
          decoration: BoxDecoration(shape: BoxShape.circle, color: color),
        ),
        const SizedBox(width: 5),
        Text(text,
            style: TextStyle(fontSize: 11, fontFamily: 'monospace',
                fontWeight: FontWeight.w700, color: color)),
      ]),
    );
  }
}

class _InfoBanner extends StatelessWidget {
  final Color color;
  final String icon;
  final String message;
  const _InfoBanner({required this.color, required this.icon, required this.message});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(children: [
        Text(icon, style: const TextStyle(fontSize: 16)),
        const SizedBox(width: 8),
        Expanded(child: Text(message,
            style: TextStyle(fontSize: 12, fontFamily: 'monospace', color: color))),
      ]),
    );
  }
}

class _MetricCell extends StatelessWidget {
  final String label, value;
  final Color color;
  const _MetricCell({required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 10),
        decoration: BoxDecoration(
          color: const Color(0x0AFFFFFF),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: const Color(0x12FFFFFF)),
        ),
        child: Column(children: [
          Text(label,
              style: const TextStyle(fontSize: 10, color: Color(0xFF6B7285),
                  fontFamily: 'monospace')),
          const SizedBox(height: 4),
          Text(value,
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700,
                  fontFamily: 'monospace', color: color)),
        ]),
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  final String title;
  const _SectionHeader({required this.title});

  @override
  Widget build(BuildContext context) {
    return Row(children: [
      Text(title.toUpperCase(),
          style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w700,
              letterSpacing: 1.5, color: Color(0xFF6B7285), fontFamily: 'monospace')),
      const SizedBox(width: 10),
      const Expanded(child: Divider(color: Color(0x12FFFFFF), thickness: 1)),
    ]);
  }
}

class _PhaseCell extends StatelessWidget {
  final WashMode mode;
  final bool isActive;
  const _PhaseCell({required this.mode, required this.isActive});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 10),
        decoration: BoxDecoration(
          color: isActive ? mode.color.withOpacity(0.1) : const Color(0xFF181D2A),
          borderRadius: BorderRadius.circular(10),
          border: Border(
            top: isActive
                ? BorderSide(color: mode.color, width: 2)
                : BorderSide.none,
            left: BorderSide(color: const Color(0x12FFFFFF)),
            right: BorderSide(color: const Color(0x12FFFFFF)),
            bottom: BorderSide(color: const Color(0x12FFFFFF)),
          ),
        ),
        child: Column(children: [
          Text(mode.label,
              style: TextStyle(
                fontSize: 12, fontWeight: FontWeight.w700,
                fontFamily: 'monospace',
                color: isActive ? mode.color : const Color(0xFF8C93A8),
              )),
          const SizedBox(height: 2),
          Text(mode.description,
              style: const TextStyle(fontSize: 10, color: Color(0xFF6B7285),
                  fontFamily: 'monospace'),
              textAlign: TextAlign.center),
        ]),
      ),
    );
  }
}

class _StatCard extends StatelessWidget {
  final String label, value, sub;
  final Color color;
  const _StatCard({required this.label, required this.value,
      required this.sub, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF181D2A),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0x12FFFFFF)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(label.toUpperCase(),
            style: const TextStyle(fontSize: 10, color: Color(0xFF6B7285),
                fontFamily: 'monospace', letterSpacing: 1)),
        const SizedBox(height: 6),
        Text(value,
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800,
                fontFamily: 'monospace', color: color)),
        Text(sub,
            style: const TextStyle(fontSize: 11, color: Color(0xFF6B7285),
                fontFamily: 'monospace')),
      ]),
    );
  }
}

class _PhaseBar extends StatelessWidget {
  final WashMode mode;
  final int count, total;
  const _PhaseBar({required this.mode, required this.count, required this.total});

  @override
  Widget build(BuildContext context) {
    final pct = total > 0 ? count / total : 0.0;
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Text(mode.label,
              style: TextStyle(fontSize: 11, fontFamily: 'monospace',
                  fontWeight: FontWeight.w700, color: mode.color)),
          const Spacer(),
          Text('$count wins · ${(pct * 100).toStringAsFixed(1)}%',
              style: const TextStyle(fontSize: 11, color: Color(0xFF6B7285),
                  fontFamily: 'monospace')),
        ]),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: pct,
            minHeight: 6,
            backgroundColor: const Color(0xFF181D2A),
            valueColor: AlwaysStoppedAnimation<Color>(mode.color),
          ),
        ),
      ]),
    );
  }
}

class _LegendDot extends StatelessWidget {
  final Color color;
  final String label;
  const _LegendDot({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Row(mainAxisSize: MainAxisSize.min, children: [
      Container(width: 16, height: 2,
          decoration: BoxDecoration(color: color,
              borderRadius: BorderRadius.circular(1))),
      const SizedBox(width: 5),
      Text(label,
          style: const TextStyle(fontSize: 10, color: Color(0xFF6B7285),
              fontFamily: 'monospace')),
    ]);
  }
}
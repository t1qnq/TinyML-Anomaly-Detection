import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:mqtt_client/mqtt_client.dart';
import 'mqtt_factory.dart'
    if (dart.library.js_interop) 'mqtt_factory_web.dart'
    as mqtt_factory;
import 'package:intl/intl.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'firebase_options.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:fl_chart/fl_chart.dart';

// ─────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────
const String kBroker    = '0a6070814ed640d2bf2200eb24c6b80e.s1.eu.hivemq.cloud';
const String kTopic = 'tinyml/quang_wm_2026/status';
const int    kPort      = 8883;
const int    kHeartbeatTimeout = 60; // seconds
const int    kAlarmWindow      = 10;  // sliding window size
const int    kAlarmEnterThr    = 5;   // 5/10 HIGH → ALARM
const int    kAlarmExitThr     = 1;   // ≤1/10 HIGH (9 OK) → thoát
const int    kMaxMaeHistory    = 40;
const int    kMaxHistory       = 30;

// MAE thresholds per mode (phải khớp firmware)
const Map<String, double> kThresholds = {
  'GENTLE': 0.1241,
  'STRONG': 0.0968,
  'SPIN':   0.0672,
};

// ─────────────────────────────────────────────
// MODELS
// ─────────────────────────────────────────────
enum MachineStatus { waiting, ok, high, alarm, deviceLost }
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
  State<MonitorScreen> createState() => _MonitorScreenStateV2();
}

class _MonitorScreenStateV2 extends State<MonitorScreen>
    with TickerProviderStateMixin {
  // MQTT
  MqttClient? _client;

  // State
  MachineStatus _status         = MachineStatus.waiting;
  WashMode      _mode           = WashMode.unknown;
  bool          _isConnected    = false;
  double        _currentMae     = 0.0;
  int           _currentWin     = 0;
  int           _currentConsec  = 0;

  // Stats (non-final for hot-reload resilience)
  int _totalWins = 0;
  int _alarmCount = 0;
  DateTime? _connectedAt;

  // History 
  List<AlarmEvent>  _events      = [];
  List<double>      _maeHistory  = [];
  List<WashMode>    _modeHistory = [];
  List<int>         _uptimeSegs  = [];

  // Phase counters
  Map<WashMode, int>    _phaseCount      = { WashMode.gentle: 0, WashMode.strong: 0, WashMode.spin: 0 };
  Map<WashMode, int>    _phaseAlarmCount = { WashMode.gentle: 0, WashMode.strong: 0, WashMode.spin: 0 };
  Map<WashMode, double> _phaseMaeSum     = { WashMode.gentle: 0.0, WashMode.strong: 0.0, WashMode.spin: 0.0 };

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
    final clientID = 'quang_wm_${DateTime.now().millisecondsSinceEpoch}';
    
    _client = mqtt_factory.createMqttClient(kBroker, clientID, kPort);

    _client!.logging(on: true); 
    _client!.setProtocolV311(); // CHÚ Ý: Bắt buộc ép chạy chuẩn 3.1.1 với HiveMQ
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

    final connMess = MqttConnectMessage()
        .withClientIdentifier(clientID)
        .authenticateAs('project1', 'Abcd2705@') // Authenticate với HiveMQ Cloud
        .startClean();
    _client!.connectionMessage = connMess;

    try {
      debugPrint('MQTT: Connecting to $kBroker...');
      await _client!.connect();
    } catch (e) {
      debugPrint('MQTT Error: $e');
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

      // Phân tích trạng thái: firmware gửi state = "OK", "HIGH", "ALARM"
      final stateStr = data['state'] as String? ?? 'OK';
      final isHigh   = stateStr == 'HIGH';
      final resolvedMode = mode != WashMode.unknown ? mode : _mode;

      setState(() {
        _totalWins++;
        _currentWin    = win;
        _currentConsec = consec;
        _currentMae    = mae;

        // MAE history + mode history (luôn đồng bộ 1:1)
        _maeHistory.add(mae);
        _modeHistory.add(resolvedMode);
        if (_maeHistory.length > kMaxMaeHistory) {
          _maeHistory.removeAt(0);
          _modeHistory.removeAt(0);
        }

        // Mode
        if (resolvedMode != WashMode.unknown) {
           _mode = resolvedMode; 
           _phaseCount[resolvedMode] = (_phaseCount[resolvedMode] ?? 0) + 1;
        }

        if (isAlarm) {
          if (_status != MachineStatus.alarm) {
            _addEvent(AlarmEvent(
              isAlarm: true, mae: mae, win: win,
              consec: consec, mode: _mode, time: DateTime.now(),
            ));
            _saveToFirestore(mae, win, consec, isAlarm: true);
            showAlarmNotification(mae);
          }
          _status = MachineStatus.alarm;
          _uptimeSegs.add(2);
          
          if (resolvedMode != WashMode.unknown) {
            _phaseAlarmCount[resolvedMode] = (_phaseAlarmCount[resolvedMode] ?? 0) + 1;
          }
          _alarmCount++; 
        } else if (isHigh) {
          _status = MachineStatus.high;
          _uptimeSegs.add(1);
        } else {
          if (_status == MachineStatus.alarm || _status == MachineStatus.high) {
            _addEvent(AlarmEvent(
              isAlarm: false, mae: mae, win: win,
              consec: consec, mode: _mode, time: DateTime.now(),
            ));
            if (_status == MachineStatus.alarm) {
              _saveToFirestore(mae, win, consec, isAlarm: false);
            }
          }
          _status = MachineStatus.ok;
          _uptimeSegs.add(0);
        }

        if (resolvedMode != WashMode.unknown) {
          _phaseMaeSum[resolvedMode] = (_phaseMaeSum[resolvedMode] ?? 0.0) + mae;
        }

        if (_uptimeSegs.length > 500) _uptimeSegs.removeAt(0);
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
      1     => _ChartTab(maeHistory: _maeHistory, modeHistory: _modeHistory, mode: _mode, phaseCount: _phaseCount, totalWins: _totalWins),
      2     => _StatsTab(
                totalWins: _totalWins, alarmCount: _alarmCount,
                connectedAt: _connectedAt, phaseCount: _phaseCount,
                uptimeSegs: _uptimeSegs, maeHistory: _maeHistory,
                phaseAlarmCount: _phaseAlarmCount,
                phaseMaeSum: _phaseMaeSum,
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
      MachineStatus.ok         => (const Color(0xFF22D9A0), '✓', 'OK'),
      MachineStatus.high       => (const Color(0xFFF5A623), '⚠', 'HIGH'),
      MachineStatus.alarm      => (const Color(0xFFFF4C6A), '⚡', 'ALARM'),
      MachineStatus.deviceLost => (const Color(0xFFF5A623), '💀', 'LOST'),
      _                        => (const Color(0xFF6B7285), '⏳', 'WAITING'),
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
                if (_status == MachineStatus.alarm || _status == MachineStatus.high)
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
          _MetricCell(label: 'HIGH/10', value: '$_currentConsec/10',
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
      clipBehavior: Clip.antiAlias,
      decoration: BoxDecoration(
        color: const Color(0xFF181D2A),
        borderRadius: BorderRadius.circular(12),
      ),
      child: IntrinsicHeight(
        child: Row(children: [
          Container(width: 3, color: color),
          Expanded(
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
          ),
        ]),
      ),
    );
  }
}

// ─────────────────────────────────────────────
// CHART TAB
// ─────────────────────────────────────────────
class _ChartTab extends StatelessWidget {
  final List<double> maeHistory;
  final List<WashMode> modeHistory;
  final WashMode mode;
  final Map<WashMode, int> phaseCount;
  final int totalWins;

  const _ChartTab({required this.maeHistory, required this.modeHistory, required this.mode, required this.phaseCount, required this.totalWins});

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
                  minY: 0, maxY: 0.35,
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
                        getDotPainter: (spot, _, __, ___) {
                          final idx = spot.x.toInt();
                          // Lấy màu theo mode tại điểm đó
                          final dotMode = (idx >= 0 && idx < modeHistory.length)
                              ? modeHistory[idx]
                              : mode;
                          // Nếu vượt ngưỡng pha hiện tại → đỏ, ngược lại → màu pha
                          final dotThr = kThresholds[dotMode.label] ?? 0.15;
                          final dotColor = spot.y > dotThr
                              ? const Color(0xFFFF4C6A)  // vượt ngưỡng → đỏ
                              : dotMode.color;           // bình thường → màu pha
                          return FlDotCirclePainter(
                            radius: 3,
                            color: dotColor,
                            strokeWidth: 0,
                          );
                        },
                      ),
                    ),
                  ],
                )),
        ),

        const SizedBox(height: 24),
        _SectionHeader(title: 'Phân phối MAE theo pha'),
        const SizedBox(height: 12),
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
// STATS TAB
// ─────────────────────────────────────────────
class _StatsTab extends StatelessWidget {
  final int totalWins;
  final int alarmCount;
  final DateTime? connectedAt;
  final Map<WashMode, int>? phaseCount;
  final Map<WashMode, int>? phaseAlarmCount;
  final Map<WashMode, double>? phaseMaeSum;
  final List<int> uptimeSegs;
  final List<double> maeHistory;

  const _StatsTab({
    required this.totalWins, required this.alarmCount,
    required this.connectedAt, required this.phaseCount,
    required this.uptimeSegs, required this.maeHistory,
    required this.phaseAlarmCount, required this.phaseMaeSum,
  });

  String get _uptime {
    if (connectedAt == null) return '—';
    final d = DateTime.now().difference(connectedAt!);
    final m = d.inMinutes, s = d.inSeconds % 60;
    return '${m}m ${s}s';
  }

  String _calcAvgMae(WashMode mode) {
    if (phaseCount == null || phaseMaeSum == null) return '—';
    final count = phaseCount![mode] ?? 0;
    if (count == 0) return '—';
    final sum = phaseMaeSum![mode] ?? 0.0;
    return (sum / count).toStringAsFixed(4);
  }

  String _calcAlarmRate(WashMode mode) {
    if (phaseCount == null || phaseAlarmCount == null) return '0.0%';
    final count = phaseCount![mode] ?? 0;
    if (count == 0) return '0.0%';
    final alarms = phaseAlarmCount![mode] ?? 0;
    return '${(alarms / count * 100).toStringAsFixed(1)}%';
  }

  String get _globalAlarmRate {
    if (totalWins == 0) return '0.0%';
    return '${(alarmCount / totalWins * 100).toStringAsFixed(1)}%';
  }

  @override
  Widget build(BuildContext context) {
    final segs = uptimeSegs.length > 500
        ? uptimeSegs.sublist(uptimeSegs.length - 500)
        : uptimeSegs;

    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        // Stat cards grid
        IntrinsicHeight(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(child: _StatCard(label: 'Tổng windows', value: '$totalWins',
                  sub: 'inference cycles', color: const Color(0xFF4A9EFF))),
              const SizedBox(width: 10),
              Expanded(child: _StatCard(
                  label: 'Số windows bị alarm', 
                  value: '$alarmCount',
                  sub: 'tỉ lệ: $_globalAlarmRate', 
                  color: const Color(0xFFFF4C6A),
                  extra: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const SizedBox(height: 8),
                      _MiniRow(label: 'GENTLE', value: _calcAlarmRate(WashMode.gentle), color: WashMode.gentle.color),
                      _MiniRow(label: 'STRONG', value: _calcAlarmRate(WashMode.strong), color: WashMode.strong.color),
                      _MiniRow(label: 'SPIN', value: _calcAlarmRate(WashMode.spin), color: WashMode.spin.color),
                    ],
                  ),
              )),
            ],
          ),
        ),
        const SizedBox(height: 10),
        IntrinsicHeight(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(child: _StatCard(
                  label: 'MAE trung bình',
                  value: '',
                  sub: 'theo từng pha', 
                  color: const Color(0xFFF5A623),
                  extra: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const SizedBox(height: 4),
                      _MiniRow(label: 'GENTLE', value: _calcAvgMae(WashMode.gentle), color: WashMode.gentle.color),
                      _MiniRow(label: 'STRONG', value: _calcAvgMae(WashMode.strong), color: WashMode.strong.color),
                      _MiniRow(label: 'SPIN', value: _calcAvgMae(WashMode.spin), color: WashMode.spin.color),
                    ],
                  ),
              )),
              const SizedBox(width: 10),
              Expanded(child: _StatCard(label: 'Uptime', value: _uptime,
                  sub: connectedAt != null
                      ? 'kể từ ${DateFormat('HH:mm').format(connectedAt!)}'
                      : 'chưa kết nối',
                  color: const Color(0xFF22D9A0))),
            ],
          ),
        ),

        const SizedBox(height: 20),

        // Uptime bar — 500 phân đoạn, 10 dòng (50/dòng)
        _SectionHeader(title: 'Trạng thái theo thời gian (${segs.length} phân đoạn)'),
        const SizedBox(height: 10),
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: const Color(0xFF111520),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: const Color(0x12FFFFFF)),
          ),
          child: segs.isEmpty
              ? const SizedBox(height: 36)
              : Column(
                  children: List.generate(10, (row) {
                    final start = row * 50;
                    final end = (start + 50).clamp(0, segs.length);
                    if (start >= segs.length) {
                      return const SizedBox.shrink();
                    }
                    final rowSegs = segs.sublist(start, end);
                    return Padding(
                      padding: EdgeInsets.only(bottom: row < 9 ? 2 : 0),
                      child: Row(
                        children: rowSegs.map((s) {
                          final c = switch (s) {
                            0 => const Color(0xFF22D9A0),  // OK = xanh
                            1 => const Color(0xFFF5A623),  // HIGH = cam
                            _ => const Color(0xFFFF4C6A),  // ALARM = đỏ
                          };
                          return Expanded(
                            child: Container(
                              height: 4, margin: const EdgeInsets.symmetric(horizontal: 0.25),
                              decoration: BoxDecoration(
                                color: c,
                                borderRadius: BorderRadius.circular(1),
                              ),
                            ),
                          );
                        }).toList(),
                      ),
                    );
                  }),
                ),
        ),

        const SizedBox(height: 20),


        
        // Technical Indicators
        _SectionHeader(title: 'Thông số kỹ thuật (V11)'),
        const SizedBox(height: 12),
        _buildTechInfo(label: 'Kiến trúc', value: 'Symmetric Autoencoder'),
        _buildTechInfo(label: 'Cấu trúc', value: '19-128-64-32-64-128-19'),
        _buildTechInfo(label: 'Tổng tham số', value: '25,779 parameters'),
        _buildTechInfo(label: 'Tối ưu hóa', value: 'INT8 QAT (Quantization)'),
        _buildTechInfo(label: 'Flash/RAM', value: '~95KB / ~120KB'),
      ],
    );
  }

  Widget _buildTechInfo({required String label, required String value}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(children: [
        Text(label, style: const TextStyle(fontSize: 12, color: Color(0xFF6B7285), fontFamily: 'monospace')),
        const Spacer(),
        Text(value, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: Color(0xFFE8EAF0), fontFamily: 'monospace')),
      ]),
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
          border: Border.all(
            color: isActive ? mode.color : const Color(0x12FFFFFF),
            width: isActive ? 2 : 1,
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
  final Widget? extra;
  const _StatCard({required this.label, required this.value,
      required this.sub, required this.color, this.extra});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF181D2A),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0x12FFFFFF)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(label.toUpperCase(),
            style: const TextStyle(fontSize: 9, color: Color(0xFF6B7285),
                fontFamily: 'monospace', letterSpacing: 1)),
        const SizedBox(height: 4),
        if (value.isNotEmpty)
          Text(value,
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800,
                  fontFamily: 'monospace', color: color)),
        Text(sub,
            style: const TextStyle(fontSize: 10, color: Color(0xFF6B7285),
                fontFamily: 'monospace')),
        if (extra != null) extra!,
      ]),
    );
  }
}

class _MiniRow extends StatelessWidget {
  final String label, value;
  final Color color;
  const _MiniRow({required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 2),
      child: Row(
        children: [
          Container(width: 4, height: 4, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
          const SizedBox(width: 6),
          Text(label, style: const TextStyle(fontSize: 9, color: Color(0xFF6B7285), fontFamily: 'monospace')),
          const Spacer(),
          Text(value, style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700, color: color, fontFamily: 'monospace')),
        ],
      ),
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
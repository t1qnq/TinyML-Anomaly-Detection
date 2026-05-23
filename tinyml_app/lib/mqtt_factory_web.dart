import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_browser_client.dart';

/// Tạo MQTT client cho bản Web dùng WebSocket Secure (WSS).
MqttClient createMqttClient(String broker, String clientId, int port) {
  // HiveMQ Cloud dùng WebSocket bảo mật ở cổng 8884 qua đường dẫn mqtt.
  final client = MqttBrowserClient('wss://$broker/mqtt', clientId);
  client.port = 8884;
  client.websocketProtocols = ['mqtt'];
  return client;
}

import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_browser_client.dart';

/// Factory tạo MQTT client — Web version (WSS)
MqttClient createMqttClient(String broker, String clientId, int port) {
  // HiveMQ Cloud WebSockets thường chạy ở cổng 8884 qua thư mục mqtt
  final client = MqttBrowserClient('wss://$broker/mqtt', clientId);
  client.port = 8884;
  client.websocketProtocols = ['mqtt'];
  return client;
}

import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_browser_client.dart';

/// Factory tạo MQTT client — Web version (WSS)
MqttClient createMqttClient(String broker, String clientId, int port) {
  final client = MqttBrowserClient('wss://$broker/mqtt', clientId);
  client.port = 8084;
  client.websocketProtocols = ['mqtt'];
  return client;
}

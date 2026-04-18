import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';

/// Factory tạo MQTT client — stub cho mobile/desktop (TCP)
MqttClient createMqttClient(String broker, String clientId, int port) {
  final client = MqttServerClient(broker, clientId);
  client.port = port;
  return client;
}

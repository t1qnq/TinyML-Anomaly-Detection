import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';

/// Tạo MQTT client dùng TLS cho các bản build mobile và desktop.
MqttClient createMqttClient(String broker, String clientId, int port) {
  final client = MqttServerClient(broker, clientId);
  client.port = port;
  client.secure = true;
  client.onBadCertificate = (dynamic cert) => true;
  return client;
}

import 'dart:io';
import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';

/// Factory tạo MQTT client — stub cho mobile/desktop (TCP)
MqttClient createMqttClient(String broker, String clientId, int port) {
  final client = MqttServerClient(broker, clientId);
  client.port = port;
  client.secure = true; // Bật TLS bảo mật
  // Bỏ qua chứng chỉ cục bộ trên App
  client.onBadCertificate = (dynamic cert) => true; 
  return client;
}

import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';
import 'dart:io';

void main() async {
  final client = MqttServerClient('0a6070814ed640d2bf2200eb24c6b80e.s1.eu.hivemq.cloud', 'test_123');
  client.port = 8883;
  client.secure = true;
  client.logging(on: true);
  client.setProtocolV311();
  final connMess = MqttConnectMessage().authenticateAs('project1', 'Abcd2705@').startClean();
  client.connectionMessage = connMess;
  client.onBadCertificate = (dynamic cert) => true; 
  try { 
      print('Connecting'); 
      await client.connect(); 
      print('Connected!'); 
  } catch(e) { 
      print(e); 
  }
  exit(0);
}

import 'dart:io';

import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';

void main() async {
  final host = Platform.environment['MQTT_HOST'];
  final username = Platform.environment['MQTT_USERNAME'];
  final password = Platform.environment['MQTT_PASSWORD'];
  final port = int.tryParse(Platform.environment['MQTT_PORT'] ?? '8883') ?? 8883;

  if (host == null || username == null || password == null) {
    stderr.writeln(
      'Set MQTT_HOST, MQTT_USERNAME, MQTT_PASSWORD and optional MQTT_PORT first.',
    );
    exitCode = 2;
    return;
  }

  final client = MqttServerClient(host, 'mqtt_scratch_test');
  client.port = port;
  client.secure = true;
  client.logging(on: true);
  client.setProtocolV311();
  client.connectionMessage = MqttConnectMessage()
      .authenticateAs(username, password)
      .startClean();
  client.onBadCertificate = (dynamic cert) => true;

  try {
    stdout.writeln('Connecting to $host:$port');
    await client.connect();
    stdout.writeln('Connected');
  } catch (error) {
    stderr.writeln(error);
    exitCode = 1;
  } finally {
    client.disconnect();
  }
}

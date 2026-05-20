import 'package:flutter_test/flutter_test.dart';
import 'package:tinyml_app/main.dart';

void main() {
  group('wash mode parsing', () {
    test('parses supported modes case-insensitively', () {
      expect(parseModeFromString('GENTLE'), WashMode.gentle);
      expect(parseModeFromString('strong'), WashMode.strong);
      expect(parseModeFromString('Spin'), WashMode.spin);
    });

    test('falls back to unknown for missing or unsupported values', () {
      expect(parseModeFromString(null), WashMode.unknown);
      expect(parseModeFromString('rinse'), WashMode.unknown);
    });
  });

  test('defines MAE thresholds for every supported wash mode', () {
    expect(kThresholds.keys, containsAll(['GENTLE', 'STRONG', 'SPIN']));
    expect(kThresholds.values.every((threshold) => threshold > 0), isTrue);
  });
}

#include <Arduino.h>

#define RELAY_PIN 27  // GPIO connected to relay IN

void setup() {
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW); // start with relay off
 Serial.begin(9600);
  Serial.println("Type 'O' to turn relay ON, 'F' to turn relay OFF");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'O' || command == 'o') {
      digitalWrite(RELAY_PIN, HIGH); // turn relay ON
      Serial.println("Relay ON");
    } 
    else if (command == 'F' || command == 'f') {
      digitalWrite(RELAY_PIN, LOW); // turn relay OFF
      Serial.println("Relay OFF");
    }
  }
}
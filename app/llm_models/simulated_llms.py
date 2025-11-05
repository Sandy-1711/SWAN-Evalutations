import time
import json

SIMULATED_RESPONSES = {
    "coder": "include <DHT.h> // Include the DHT library\n\n#define DHTPIN 4    // Pin connected to DHT11 data pin\n#define DHTTYPE DHT11  // Type of DHT sensor (DHT11, DHT22, or DHT12)\n#define LEDPIN 7    // Pin connected to the LED\n\nDHT dht(DHTPIN, DHTTYPE); // Create a DHT object\n\nvoid setup() {\n  Serial.begin(9600); // Initialize serial communication\n  dht.begin();       // Initialize the DHT sensor\n  pinMode(LEDPIN, OUTPUT); // Set the LED pin as an output\n  digitalWrite(LEDPIN, LOW);  // Ensure the LED is off initially\n}\n\nvoid loop() {\n  // Read humidity from the DHT11 sensor\n  float humidity = dht.readHumidity();\n\n  // Check if reading was successful\n  if (isnan(humidity)) {\n    Serial.println(\"Failed to read from DHT sensor!\"); // Print error message\n  } else {\n    Serial.print(\"Humidity: \");\n    Serial.print(humidity);\n    Serial.println(\" %\"); // Print humidity value\n\n    // Check if humidity is greater than 70%\n    if (humidity > 70) {\n      digitalWrite(LEDPIN, HIGH); // Turn the LED on\n      Serial.println(\"Humidity is high! LED is ON\");\n    } else {\n      digitalWrite(LEDPIN, LOW);  // Turn the LED off\n      Serial.println(\"Humidity is normal. LED is OFF\");\n    }\n  }\n  delay(2000); // Wait 2 seconds before the next reading\n}",
    "compressor": "<<=components=>>\nesp:wokwi-esp8266\ndht:wokwi-dht11\nled1:wokwi-led\nbb1:wokwi-breadboard\n<<=connections=>>\nesp:3V3 bb1:tp.36\nesp:GND.1 bb1:tn.40\ndht:VCC bb1:tp.11\ndht:SDA bb1:19t.b\ndht:GND bb1:tn.16\nesp:GND.2 bb1:bn.42\nled1:C bb1:bn.41\nled1:A bb1:38t.b\nesp:D7 bb1:38t.e\nesp:D4 bb1:19t.e\n<<=attrs=>>\nled1 color:red",
    "generator": json.dumps({
      "parts": [
        {
          "id": "uno",
          "type": "wokwi-arduino-uno",
          "attrs": {}
        },
        {
          "id": "bb1",
          "type": "wokwi-breadboard",
          "attrs": {}
        },
        {
          "id": "led1",
          "type": "wokwi-led",
          "attrs": {
            "color": "red"
          }
        },
        {
          "id": "r1",
          "type": "wokwi-resistor",
          "attrs": {
            "value": "220"
          }
        },
        {
          "id": "sensor1",
          "type": "wokwi-digital-motion-sensor",
          "attrs": {}
        }
      ],
      "connections": [
        [
          "bb1:10t.a",
          "led1:A",
          "red",
          [
            "v0"
          ]
        ],
        [
          "bb1:11t.a",
          "led1:C",
          "black",
          [
            "v0"
          ]
        ],
        [
          "r1:1",
          "bb1:10t.b",
          "red",
          [
            "v0"
          ]
        ],
        [
          "r1:2",
          "bb1:11t.b",
          "black",
          [
            "v0"
          ]
        ],
        [
          "uno:GND.2",
          "bb1:11t.e",
          "black",
          [
            "v0"
          ]
        ],
        [
          "sensor1:VCC",
          "bb1:20t.e",
          "red",
          [
            "v0"
          ]
        ],
        [
          "sensor1:GND",
          "bb1:21t.e",
          "black",
          [
            "v0"
          ]
        ],
        [
          "sensor1:OUT",
          "bb1:21t.a",
          "green",
          [
            "v0"
          ]
        ],
        [
          "bb1:21t.d",
          "uno:2",
          "green",
          [
            "v0"
          ]
        ],
        [
          "uno:GND.1",
          "bb1:21t.c",
          "black",
          [
            "v0"
          ]
        ]
      ]
    }),
    "baseline": "```cpp\ninclude <DHT.h> // Include the DHT library\n\n#define DHTPIN 4    // Pin connected to DHT11 data pin\n#define DHTTYPE DHT11  // Type of DHT sensor (DHT11, DHT22, or DHT12)\n#define LEDPIN 7    // Pin connected to the LED\n\nDHT dht(DHTPIN, DHTTYPE); // Create a DHT object\n\nvoid setup() {\n  Serial.begin(9600); // Initialize serial communication\n  dht.begin();       // Initialize the DHT sensor\n  pinMode(LEDPIN, OUTPUT); // Set the LED pin as an output\n  digitalWrite(LEDPIN, LOW);  // Ensure the LED is off initially\n}\n\nvoid loop() {\n  // Read humidity from the DHT11 sensor\n  float humidity = dht.readHumidity();\n\n  // Check if reading was successful\n  if (isnan(humidity)) {\n    Serial.println(\"Failed to read from DHT sensor!\"); // Print error message\n  } else {\n    Serial.print(\"Humidity: \");\n    Serial.print(humidity);\n    Serial.println(\" %\"); // Print humidity value\n\n    // Check if humidity is greater than 70%\n    if (humidity > 70) {\n      digitalWrite(LEDPIN, HIGH); // Turn the LED on\n      Serial.println(\"Humidity is high! LED is ON\");\n    } else {\n      digitalWrite(LEDPIN, LOW);  // Turn the LED off\n      Serial.println(\"Humidity is normal. LED is OFF\");\n    }\n  }\n  delay(2000); // Wait 2 seconds before the next reading\n}```\n```json{\"parts\":[{\"id\":\"nano\",\"type\":\"wokwi-arduino-nano\"},{\"id\":\"pir\",\"type\":\"wokwi-pir-motion-sensor\"},{\"id\":\"mq2\",\"type\":\"wokwi-gas-sensor\"},{\"id\":\"bb1\",\"type\":\"wokwi-breadboard\"}],\"connections\":[[\"nano:A0\",\"bb1:22t.e\",\"green\",[\"v0\"]],[\"mq2:AOUT\",\"bb1:9t.a\",\"green\",[\"v0\"]],[\"pir:OUT\",\"bb1:29t.a\",\"green\",[\"v0\"]],[\"bb1:29t.e\",\"nano:2\",\"green\",[\"v0\"]]]}```",
    "base": "HEHEHE",
}


class SimulatedLlama:
    def __init__(self, model_name):
        self.model_name = model_name
        self.response_text = SIMULATED_RESPONSES.get(model_name, "[Simulation Missing]")

    def __call__(self, prompt: str, max_tokens=1024, stream=False, stop=None):
        if not stream:
            return {
                "choices": [
                    {
                        "text": self.response_text
                    }
                ]
            }
        else:
            return self._stream_response()

    def _stream_response(self):
        for token in self.response_text.split():
            yield {"choices": [{"text": token + " "}]}
            time.sleep(0.02)  # Optional: mimic generation delay

coder_llm = SimulatedLlama("coder")
compressor_llm = SimulatedLlama("compressor")
generator_llm = SimulatedLlama("generator")
baseline_llm = SimulatedLlama("baseline")
base_llm = SimulatedLlama("base")

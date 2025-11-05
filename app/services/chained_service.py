from app.llm_models.shared_llms import coder_llm_2048, compressor_llm_2048, generator_llm_2048

model = None
tok = None


CODER_PROMPT = """You are an expert IoT code generation engine. Your sole purpose is to convert a user request into a single, complete, and functional block of code for the specified microcontroller.

    **User Request:** <<< {user_prompt} >>>

    **KEY INSTRUCTIONS:**
    1.  **Complete Requirement Fulfillment:** Your code MUST implement every feature, sensor, and logic step mentioned in the user request.
    2.  **Library and API Precision (CRITICAL):** You MUST use the exact, correct libraries and function calls for the specified hardware and components. For example, use `Adafruit_BME280.h` for a BME280 sensor or `WiFi.h` for an ESP32. Do not use placeholder or generic libraries.
    3.  **Self-Contained Code:** Generate a single, complete code file. It must include all necessary parts: library includes, variable/object declarations, a full `setup()` function, and a full `loop()` function containing the main logic.

    **OUTPUT MANDATE:**
    -   **Return ONLY raw source code.**
    -   Your response MUST NOT contain any explanations, comments, markdown, or any text other than the code itself.
    -   The first line of your output must be the first line of the code (e.g., an `#include` statement). """

COMPRESSOR_PROMPT = (
    "You are an **experienced Arduino Systems Engineer**.\n"
    "Given:\n"
    "  • User prompt: {user_prompt}\n"
    "  • Arduino code:\n{code}\n\n"
    "Generate a **compressed hardware spec** in *Wokwi* nomenclature using **_exactly_** "
    "this scaffold (do not add / remove headers or blank lines):\n"
    "<<=components=>>\n"
    "<<=connections=>>\n"
    "<<=attrs=>>\n\n"
    "• **components** - one per line as `<id>:<wokwi-part-id>`\n"
    "• **connections** - one per line as `<src> <dst>`\n"
    "• **attrs** - optional key-value extras, one per line as `<id> <key>:<value>`\n"
    "Capture every pin / wiring detail needed to reproduce the circuit, "
    "omit text that is not required for the diagram.\n "
)

EXAMPLE_JSON = (
    '{"parts":[{"id":"esp","type":"wokwi-esp8266"},'
    '{"id":"dht","type":"wokwi-dht11"},'
    '{"id":"led1","type":"wokwi-led","attrs":{"color":"red"}},'
    '{"id":"bb1","type":"wokwi-breadboard"}],'
    '"connections":[["esp:3V3","bb1:tp.36","red",["v0"]]]}'
)

GENERATOR_PROMPT = (
    "You are an **experienced Arduino Systems Engineer**.\n"
    "Given the following circuit specification, produce **only** a JSON object\n"
    "with *two* top‑level keys: `parts` (array) and `connections` (array).\n"
    "Each element of `parts` must have `id`, `type` and optional `attrs`.\n"
    "Each element of `connections` must be an array⃰ of the form\n"
    "[from, to, color, path]. Use the same IDs as in `parts`.\n\n"
    "Return strictly valid JSON — no markdown, code fences, or commentary.\n"
    "If any attribute is missing, infer sensible defaults.\n\n"
    "### Example format\n" + EXAMPLE_JSON.replace("{", "{{").replace("}", "}}") + "\n\n"
    "Now read the specification and output the JSON: \n"
    "{specification} "
)


def generate_code(llm, prompt: str) -> str:
    system_message = CODER_PROMPT.format(user_prompt=prompt.strip())
    chat_input = f"<|im_start|>system\n{system_message.strip()}\n<|im_end|>\n<|im_start|>assistant\n"

    output = llm(
        chat_input,
        max_tokens=1024,
        stop=["<|im_end|>"],
    )
    return output.get("choices", [{}])[0].get("text", "").strip()


def compress_to_ir(llm, prompt: str, code: str) -> str:
    message = COMPRESSOR_PROMPT.format(user_prompt=prompt, code=code).strip()
    user_block = f"{prompt}\n\n{code}"
    chat_input = (
        f"<|im_start|>system\n{message}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_block}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    response = llm(
        chat_input,
        max_tokens=1024,
        stop=["<|im_end|>"],
    )
    return response.get("choices", [{}])[0].get("text", "").strip()


def generate_json(llm, specification: str) -> str:
    message = GENERATOR_PROMPT.format(specification=specification).strip()
    chat_input = f"<|im_start|>system\n{message}\n<|im_end|>\n<|im_start|>assistant\n"
    print(chat_input)
    response = llm(
        chat_input,
        max_tokens=1024,
        stop=["<|im_end|>"],
    )
    return response.get("choices", [{}])[0].get("text", "").strip()


def run_pipeline(user_prompt: str):
    code = generate_code(coder_llm_2048, user_prompt)
    ir = compress_to_ir(compressor_llm_2048, user_prompt, code)
    json_output = generate_json(generator_llm_2048, ir)

    return code, ir, json_output

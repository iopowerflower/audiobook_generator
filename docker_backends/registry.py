"""
Registry of known Docker TTS backends.

Each backend can be started/stopped from the UI. Voices from running
backends appear in the Docker TTS tab.
"""

KNOWN_BACKENDS = [
    {
        "id": "kani-pt",
        "name": "Kani TTS (Multilingual)",
        "image": "registry.hf.space/nineninesix-kani-tts-2-pt:latest",
        "hf_port": 7860,
        "proxy_port": 7862,
        "hf_container": "kani-hf",
        "proxy_container": "kani-proxy",
        "proxy_dir": "kani_pt",
        "proxy_image": "audiobook-kani-pt-proxy:latest",
    },
    {
        "id": "kani-en",
        "name": "Kani TTS (English Accents)",
        "image": "registry.hf.space/nineninesix-kanitts-2-en:latest",
        "hf_port": 7864,
        "proxy_port": 7866,
        "hf_container": "kani-en-hf",
        "proxy_container": "kani-en-proxy",
        "proxy_dir": "kani_en",
        "proxy_image": "audiobook-kani-en-proxy:latest",
    },
]

import base64

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from safe_call import safe_call

load_dotenv()


# ==========================================
# Vision Model
# ==========================================

VISION_MODEL = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct"
)


# ==========================================
# Image Encoder
# ==========================================

def encode_image(image_path: str):

    with open(image_path, "rb") as image_file:

        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


# ==========================================
# Vision Tool
# ==========================================

@safe_call("Vision Tool")
def describe_image(image_path: str):

    """
    Analyses an uploaded image
    using Groq Vision.
    """

    encoded_image = encode_image(
        image_path
    )

    message = [

        {
            "role": "user",

            "content": [

                {
                    "type": "text",

                    "text":
                    """
Describe this image carefully.

Mention:

• Main objects

• Text visible

• Charts/Tables (if any)

• Scene

• Important details
"""
                },

                {
                    "type": "image_url",

                    "image_url": {
                        "url":
                        f"data:image/jpeg;base64,{encoded_image}"
                    }
                }

            ]

        }

    ]

    response = VISION_MODEL.invoke(
        message
    )

    return response.content
import asyncio
from lmmule.mule import ALLOWED_TAG_DEFAULT, Mule


async def main():
    page_content = (
        await Mule.scrape_page(
            title="Keras Custom",
            url="https://keras.io/guides/making_new_layers_and_models_via_subclassing/",
            # url="https://stackoverflow.com/questions/4681317/in-lxml-how-do-i-remove-a-tag-but-retain-all-contents",
            allowed_tags=ALLOWED_TAG_DEFAULT,
        )
    ).get("content", "")
    print(page_content)


if __name__ == "__main__":
    asyncio.run(main())

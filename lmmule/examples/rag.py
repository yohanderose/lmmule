import asyncio
import os
import json

from lmmule.rag import Rag, OllamaEmbedding


async def main():
    rag = await Rag(
        postgres_url=f'postgresql+asyncpg://postgres:{os.environ.get("PG_PASS")}@localhost:5432/veda',
        embedder=OllamaEmbedding(model_name="embeddinggemma", embed_dim=768),
    ).init_db()

    docs = [
        "Indigestion stems from weak digestive fire (agni) and ama, often from Kapha or Vata excess. Ginger, fennel, and triphala kindle agni, paired with warm, light foods avoiding cold items. Abdominal massage and yoga twists aid relief.​",
        "Cold and cough arise from Vata-Kapha imbalance due to exposure or low immunity. Tulsi, ginger, pippali, and licorice clear symptoms, with warm soups, honey, and no dairy. Steam inhalation and pranayama support recovery.​",
        "Arthritis results from Vata aggravation causing joint pain and stiffness. Guggulu, ashwagandha, and turmeric reduce inflammation, alongside oil massages and enemas. Warm, Vata-pacifying foods like sesame oil help.​",
        "Fever involves Pitta-Kapha with ama toxins. Guduchi, neem, and chirayata clear heat, using light barley water and pomegranate. Fasting and cooling therapies like purgation follow.​",
        "Eczema and similar conditions stem from Pitta or Kapha with impure blood. Neem, manjistha, and turmeric detoxify, via bloodletting or pastes. Bitter greens and avoiding spicy foods aid healing.​",
        "Asthma occurs when Kapha blocks lung channels. Vasa, kantakari, and pippali open airways, with steam and pranayama. Avoiding cold environments and using garlic helps.​",
        "Hypertension reflects Pitta-Vata stress on channels. Arjuna, ashwagandha, and brahmi calm, with cooling fruits and low salt. Yoga, meditation, and Shirodhara provide relief.​",
        "Diabetes involves Kapha-Pitta impairing fat metabolism. Bitter melon, fenugreek, and guduchi regulate, using low-sweet, high-fiber diets. Exercise and yoga balance agni.​",
        "Insomnia arises from Vata disturbing the mind. Jatamansi, shankhpushpi, and tagara soothe, with warm foot massages. Early dinners and bedtime meditation restore sleep",
    ]

    for s, ds, n in [
        (
            "Charaka Saṃhitā",
            docs[:3],
            "user1",
        ),
        (
            "Charaka Saṃhitā",
            docs[:3],
            "user2",
        ),
        (
            "Suśruta Saṃhitā",
            docs[3:6],
            "user1",
        ),
        (
            "Aṣṭāṅga Hṛdayam ",
            docs[6:],
            "user1",
        ),
    ]:
        s_id = await rag.upsert_source(s)
        await rag.upsert_documents(texts=ds, source_id=s_id, namespace=n)

    res = await rag.search(query="body ache", namespace="user1")
    print(json.dumps(res, indent=2))

    res = await rag.get_all(namespace="user2")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

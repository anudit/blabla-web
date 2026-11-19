/**
 * English voices for Mach Boom from gradium/phonon-voices (Apache-2.0).
 * Dependency-free so the UI can import it without pulling in the engine.
 */
export const VOICE_BASE_URL = 'https://huggingface.co/gradium/phonon-voices/resolve/main/voices';

export const MACH_BOOM_VOICES: readonly { name: string; id: string; label: string }[] = [
  { name: "Zoey", id: "NbpkqMVS3CJeq2j8", label: 'Zoey (US F)' },
  { name: "Juniper", id: "NIWIOv5vTi5eRMj8", label: 'Juniper (US F)' },
  { name: "Everly", id: "vjFyp2BcCyTlUmYb", label: 'Everly (US F)' },
  { name: "Harper", id: "4SZHfMpw-p46Ywgs", label: 'Harper (US F)' },
  { name: "Hadley", id: "bXwrkHhURL1VFJdG", label: 'Hadley (US F)' },
  { name: "Delia", id: "nGzMHMPAWGaJFHOo", label: 'Delia (US F)' },
  { name: "Easton", id: "LMDIYE0Oa377Qh6Z", label: 'Easton (US M)' },
  { name: "Micah", id: "40QrMVgAxFiUnP9f", label: 'Micah (US M)' },
  { name: "Trevor", id: "0zLApsZfVLLNaEGl", label: 'Trevor (US M)' },
  { name: "Russell", id: "_6Aslh2DxfmnRLmP", label: 'Russell (US M)' },
  { name: "Preston", id: "qIxlZ7Ei8KvWlyQH", label: 'Preston (US M)' },
  { name: "Miles", id: "d9XCyhjo4JhkHF1d", label: 'Miles (US M)' },
  { name: "Maeve", id: "6PWnV0Nq4wu7RVBT", label: 'Maeve (UK F)' },
  { name: "Freya", id: "GgfEkEJtxZR7gnpy", label: 'Freya (UK F)' },
  { name: "Pippa", id: "uem82D50GRv2Dwma", label: 'Pippa (UK F)' },
  { name: "Elodie-Rose", id: "y9bQqIDnyxwqK01t", label: 'Elodie-Rose (UK F)' },
  { name: "Imogen", id: "gDT1nz7Ie36ZhL-C", label: 'Imogen (UK F)' },
  { name: "Lottie", id: "m9MxpY6IUGXtWp76", label: 'Lottie (UK F)' },
  { name: "Alfie", id: "X6NceqF285Gr04gb", label: 'Alfie (UK M)' },
  { name: "Hugh", id: "YyKcxPG7iIgS3jsK", label: 'Hugh (UK M)' },
  { name: "Edwin", id: "zUCxzrOrugTSis03", label: 'Edwin (UK M)' },
  { name: "Spencer", id: "JOW0vchyswkQUMDS", label: 'Spencer (UK M)' },
  { name: "Toby", id: "dME3IWyZBvmh1n1q", label: 'Toby (UK M)' },
  { name: "Teddy", id: "Zn7g6NxLqXlJb7Qi", label: 'Teddy (UK M)' },
  { name: "Bryony", id: "gfP44RlHwr2kEwk7", label: 'Bryony (UK F)' },
];

export const DEFAULT_VOICE = 'Zoey';

/** Voice name -> file id (`${VOICE_BASE_URL}/${id}.safetensors`). */
export const VOICE_IDS: Record<string, string> = Object.fromEntries(MACH_BOOM_VOICES.map((v) => [v.name, v.id]));

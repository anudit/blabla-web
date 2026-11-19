import { signal } from '@preact/signals';

export const isPlayingSignal            = signal(false);
export const playbackStateSignal        = signal<string>("Idle");
export const ttsStatusSignal            = signal<string>("Init");
export const isModelReadySignal         = signal(false);
export const currentSentenceIndexSignal = signal(-1);
export const restartSignal              = signal(0);

// Document State
export const sentencesSignal            = signal<any[]>([]);
export const fileTypeSignal             = signal<'pdf' | 'epub' | 'text' | 'ocr' | null>(null);
export const playbackSpeedSignal        = signal(1.0);
export const selectedVoiceSignal        = signal('Zoey');
export const volumeSignal                = signal(0.8);
export const currentFileIdSignal        = signal<string | null>(null);
export const currentFileNameSignal      = signal<string | null>(null);
export const outlineSignal              = signal<any[]>([]);
export const offlineReadySignal         = signal(false);
export const pipActiveSignal            = signal(false);

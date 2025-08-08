import { getRequestConfig } from 'next-intl/server';

// Only English is supported
export const locales = ['en'];

export default getRequestConfig(async ({ locale }) => {
  // Always use English - other languages removed
  const safeLocale = 'en';

  return {
    locale: safeLocale,
    messages: (await import(`./messages/en.json`)).default
  };
});

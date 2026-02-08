const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  const url = 'https://appemergenciapy-evagabxskaw7cqvwex2d3k.streamlit.app/';
  
  console.log(`Visitando ${url}...`);
  await page.goto(url, { waitUntil: 'networkidle' });

  // Intentamos localizar el botón de "Wake up"
  // Streamlit suele usar un botón que contiene el texto "Wake up"
  const wakeUpButton = page.locator('button:has-text("Wake up")');

  if (await wakeUpButton.isVisible()) {
    console.log('La app está dormida. Haciendo clic en "Wake up"...');
    await wakeUpButton.click();
    // Esperamos un poco a que empiece a cargar
    await page.waitForTimeout(5000); 
    console.log('¡Clic realizado con éxito!');
  } else {
    console.log('La app ya está despierta o el botón no es visible.');
  }

  await browser.close();
})();

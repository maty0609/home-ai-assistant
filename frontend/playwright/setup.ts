import { _electron as electron, _electron as electronMain } from 'playwright';
import { chromium } from 'playwright';

export async function setup() {
  // This setup file runs before all tests
  // You can configure the browser or perform setup tasks here
  console.log('Playwright tests are running');
}

export async function teardown() {
  // Cleanup after all tests
  console.log('Playwright tests completed');
}

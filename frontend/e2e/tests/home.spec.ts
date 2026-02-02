import { test, expect } from '@playwright/test';

test('homepage loads successfully', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: /Marvin/i })).toBeVisible();
});

test('homepage displays user information', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: /Marvin/i })).toBeVisible();
});

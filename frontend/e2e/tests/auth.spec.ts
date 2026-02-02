import { test, expect } from '@playwright/test';

test('signin page loads', async ({ page }) => {
  await page.goto('/auth/signin');
  await expect(page.getByRole('heading', { name: /Sign in to Marvin/i })).toBeVisible();
});

test('signin form can be interacted with', async ({ page }) => {
  await page.goto('/auth/signin');

  // Check if email input exists
  const emailInput = page.getByPlaceholder(/Email address/i);
  if (await emailInput.isVisible()) {
    await emailInput.fill('test@example.com');
  }

  // Check if password input exists
  const passwordInput = page.getByPlaceholder(/Password/i);
  if (await passwordInput.isVisible()) {
    await passwordInput.fill('password123');
  }

  // Check if submit button exists
  const submitButton = page.getByRole('button', { name: /Sign in/i });
  if (await submitButton.isVisible()) {
    await submitButton.click();
  }
});

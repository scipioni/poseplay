### Workflow: OpenSpec Refine Change Proposal

This workflow is designed to update the detailed specifications (spec deltas) and the implementation task list (tasks.md) within an existing OpenSpec change folder after the main proposal.md has been modified.

**Input:**
- A valid OpenSpec change ID (e.g., 'add-user-profiles').

**Steps:**
1. **Analyze Proposal:** Read the contents of the `openspec/changes/<change-id>/proposal.md` file to understand the new or modified requirements.
2. **Update Spec Delta:** Using the new requirements from the proposal, carefully modify the spec delta files (located in `openspec/changes/<change-id>/specs/`) to include, modify, or remove requirements and scenarios.
3. **Update Tasks:** Review and update the `openspec/changes/<change-id>/tasks.md` file, ensuring the checklist of implementation steps is accurate, complete, and aligned with the newly refined spec delta.
4. **Validation (Optional):** After updating, run the `openspec validate <change-id>` command in the terminal to check for any structural errors in the spec deltas. Inform the user of the validation result.
5. **Confirmation:** Confirm to the user that the specs and tasks have been successfully refined based on the proposal updates.

**Goal:** Synchronize the spec deltas and implementation tasks with the latest version of the change proposal.

import { render, fireEvent, screen, waitFor } from '@testing-library/react';
import { vi, describe, it, expect } from 'vitest';
import IngestFileUploader from '@/components/IngestFileUploader';
import * as apiModule from '@/services/api';

describe('IngestFileUploader', () => {
  it('uploads file and shows success message', async () => {
    const fakeResponse = { success: true, node_id: '123', embedding_dimension: 8 };

    const spy = vi.spyOn(apiModule.api, 'ingestFile').mockResolvedValue(fakeResponse as any);

    render(<IngestFileUploader />);

    const namespace = screen.getByTestId('namespace-input') as HTMLInputElement;
    const tags = screen.getByTestId('tags-input') as HTMLInputElement;
    const upload = screen.getByTestId('upload-button');

    // set namespace and tags
    fireEvent.change(namespace, { target: { value: 'test_ns' } });
    fireEvent.change(tags, { target: { value: 'a,b' } });

    // attach a file via the hidden input
    const fileInput = screen.getByTestId('file-input') as HTMLInputElement;

    const file = new File(['hello world'], 'hello.txt', { type: 'text/plain' });
    // simulate user selecting a file
    fireEvent.change(fileInput, { target: { files: [file] } });

    // click upload
    fireEvent.click(upload);

    await waitFor(() => expect(spy).toHaveBeenCalled());

    const msg = await screen.findByTestId('message');
    expect(msg.textContent).toContain('uploaded');

    spy.mockRestore();
  });
});

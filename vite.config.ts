import { createReadStream, readdirSync, realpathSync, statSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, loadEnv, type Connect, type Plugin } from 'vite';

const project = fileURLToPath(new URL('.', import.meta.url));

function localPath(env: Record<string, string>, key: string, fallback: string): string {
  return path.resolve(project, env[key] || fallback);
}

function requestedFile(request: IncomingMessage, response: ServerResponse): string | null {
  try { return decodeURIComponent((request.url ?? '').split('?')[0].slice(1)); }
  catch { response.statusCode = 400; response.end('Invalid asset path'); return null; }
}

function availableModels(directory: string, configured: string): string[] {
  try {
    const allowed = configured ? new Set(configured.split(',').map(name => name.trim())) : null;
    return readdirSync(directory).filter(name => /[.]vrm$/i.test(name)
      && (!allowed || allowed.has(name)) && statSync(path.join(directory, name)).isFile()).sort();
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return [];
    throw error;
  }
}

function streamAsset(directory: string, filename: string, contentType: string,
  response: ServerResponse, next: Connect.NextFunction): void {
  try {
    const base = realpathSync(directory), filepath = realpathSync(path.join(base, filename));
    const relative = path.relative(base, filepath);
    if (relative.startsWith('..') || path.isAbsolute(relative) || !statSync(filepath).isFile()) {
      response.statusCode = 403; response.end('Asset path rejected'); return;
    }
    response.setHeader('Content-Type', contentType);
    response.setHeader('Content-Length', statSync(filepath).size);
    createReadStream(filepath).on('error', error => response.destroy(error)).pipe(response);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      response.statusCode = 404; response.end('Optional local asset is not installed'); return;
    }
    next(error instanceof Error ? error : new Error('Unable to read local asset'));
  }
}

function serveAssets(env: Record<string, string>): Plugin {
  const modelDirectory = localPath(env, 'MOTION_VRM_DIR', 'models/avatars');
  const ardyDirectory = localPath(env, 'MOTION_ARDY_ASSETS_DIR', 'models/ardy');
  const lumenDirectory = localPath(env, 'MOTION_LUMEN_DIR', 'external/HAST-SEC');
  const register = (server: { middlewares: Pick<Connect.Server, 'use'> }): void => {
    server.middlewares.use('/api/models', (_request, response, next) => {
      try {
        const files = availableModels(modelDirectory, env.MOTION_VRM_FILES || '');
        const models = files.map(file => ({ file, name: path.parse(file).name.replace(/[<>"'&]/g, ''),
          size: `${Math.round(statSync(path.join(modelDirectory, file)).size / 1048576)} MB` }));
        response.setHeader('Content-Type', 'application/json'); response.end(JSON.stringify(models));
      } catch (error) { next(error instanceof Error ? error : new Error('Unable to list VRM files')); }
    });
    server.middlewares.use('/models', (request, response, next) => {
      const file = requestedFile(request, response); if (file === null) return;
      if (!availableModels(modelDirectory, env.MOTION_VRM_FILES || '').includes(file)) {
        response.statusCode = 404; response.end('Model not found'); return;
      }
      streamAsset(modelDirectory, file, 'model/gltf-binary', response, next);
    });
    server.middlewares.use('/ardy', (request, response, next) => {
      const file = requestedFile(request, response); if (file === null) return;
      if (!['manifest.json', 'ardy_s_curve.json', 'ardy_walk_long.json'].includes(file)) {
        response.statusCode = 404; response.end('Motion not found'); return;
      }
      streamAsset(ardyDirectory, file, 'application/json', response, next);
    });
    server.middlewares.use('/external/HAST-SEC', (request, response, next) => {
      const file = requestedFile(request, response); if (file === null) return;
      if (!/^models\/skills\/(manifest|class-\d+|sources-[0-3])[.]json$/.test(file)
        && !/^models\/trajectories\/[\w-]+[.]bin$/.test(file)) {
        response.statusCode = 404; response.end('Lumen asset not found'); return;
      }
      streamAsset(lumenDirectory, file, file.endsWith('.json') ? 'application/json' : 'application/octet-stream', response, next);
    });
  };
  return { name: 'roamind-local-assets', configureServer: register, configurePreviewServer: register };
}

export default defineConfig(({ mode }) => ({
  plugins: [serveAssets({ ...loadEnv(mode, project, ''), ...process.env } as Record<string, string>)],
  build: { target: 'es2022', rollupOptions: { input: { main: path.join(project, 'index.html'), comparison: path.join(project, 'hast-comparison.html') } } },
}));

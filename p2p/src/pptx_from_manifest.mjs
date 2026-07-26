#!/usr/bin/env node

import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";


// The bundled helper uses HOME on every platform; Windows normally exposes
// USERPROFILE instead. Set the conventional alias before importing the helper.
if (!process.env.HOME) process.env.HOME = os.homedir();


function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const next = argv[index + 1];
    if (!next || next.startsWith("--")) {
      args[token.slice(2)] = true;
    } else {
      args[token.slice(2)] = next;
      index += 1;
    }
  }
  return args;
}


function requireArg(args, key) {
  if (!args[key] || typeof args[key] !== "string") throw new Error(`Missing --${key}`);
  return path.resolve(args[key]);
}


function pad(value) {
  return String(value).padStart(2, "0");
}


function positionFor(item, slideSpec, slideSize) {
  const scaleX = slideSize.width / Number(slideSpec.sourceWidth);
  const scaleY = slideSize.height / Number(slideSpec.sourceHeight);
  return {
    left: Number(item.x) * scaleX,
    top: Number(item.y) * scaleY,
    width: Math.max(1, Number(item.w) * scaleX),
    height: Math.max(1, Number(item.h) * scaleY),
  };
}


function normalizeColor(value, fallback = "#000000") {
  if (!value) return fallback;
  const text = String(value).trim();
  if (/^#[0-9a-f]{6,8}$/i.test(text)) return text;
  if (/^[0-9a-f]{6,8}$/i.test(text)) return `#${text}`;
  return fallback;
}


async function findArtifactUtils() {
  const candidates = [];
  if (process.env.PRESENTATIONS_ARTIFACT_UTILS) {
    candidates.push(process.env.PRESENTATIONS_ARTIFACT_UTILS);
  }
  const versionsRoot = path.join(
    os.homedir(),
    ".codex",
    "plugins",
    "cache",
    "openai-primary-runtime",
    "presentations",
  );
  try {
    const entries = await fs.readdir(versionsRoot, { withFileTypes: true });
    const versions = entries
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .sort()
      .reverse();
    for (const version of versions) {
      const skillRoot = path.join(
        versionsRoot,
        version,
        "skills",
        "presentations",
      );
      candidates.push(path.join(skillRoot, "container_tools", "artifact_tool_utils.mjs"));
      candidates.push(path.join(skillRoot, "scripts", "artifact_tool_utils.mjs"));
    }
  } catch {
    // A public PptxGenJS fallback is used outside a Codex presentations runtime.
  }
  for (const candidate of candidates) {
    try {
      await fs.access(candidate);
      return await import(pathToFileURL(path.resolve(candidate)).href);
    } catch {
      // Try the next installed runtime.
    }
  }
  return null;
}


function loadPptxGenJS() {
  const localRequire = createRequire(import.meta.url);
  try {
    const module = localRequire("pptxgenjs");
    return module.default || module;
  } catch {
    const bundledPath = path.join(
      os.homedir(),
      ".cache",
      "codex-runtimes",
      "codex-primary-runtime",
      "dependencies",
      "node",
      "node_modules",
      "pptxgenjs",
    );
    try {
      const module = localRequire(bundledPath);
      return module.default || module;
    } catch (error) {
      throw new Error(
        `PptxGenJS is unavailable. Run "npm install" in p2p/src. ${error.message}`,
      );
    }
  }
}


async function addArtifactImage(slide, utils, file, position, name) {
  const image = slide.images.add({
    blob: await utils.readImageBlob(file),
    fit: "cover",
    alt: name,
    name,
  });
  image.position = position;
}


async function buildWithArtifact({ manifest, out, workspace, previewDir, layoutDir, utils }) {
  await utils.ensureArtifactToolWorkspace(workspace);
  const artifact = await utils.importArtifactTool(workspace);
  const { Presentation, PresentationFile } = artifact;
  const slideSize = manifest.slideSize;
  const presentation = Presentation.create({ slideSize });
  const slideRecords = [];

  for (const slideSpec of manifest.slides) {
    const slide = presentation.slides.add();
    slide.background.fill = "#FFFFFF";
    await addArtifactImage(
      slide,
      utils,
      slideSpec.background,
      { left: 0, top: 0, width: slideSize.width, height: slideSize.height },
      `slide-${pad(slideSpec.number)}-background`,
    );

    for (const item of slideSpec.images || []) {
      await addArtifactImage(
        slide,
        utils,
        item.file,
        positionFor(item, slideSpec, slideSize),
        item.name,
      );
    }

    for (const item of slideSpec.shapes || []) {
      const shape = slide.shapes.add({
        geometry: item.kind || "rect",
        name: item.name,
        position: positionFor(item, slideSpec, slideSize),
        fill: normalizeColor(item.fill, "#00000000"),
        line: {
          style: "solid",
          fill: normalizeColor(item.line, "#00000000"),
          width: Number(item.lineWidth || 0),
        },
      });
      shape.text = "";
    }

    for (const item of slideSpec.texts || []) {
      const position = positionFor(item, slideSpec, slideSize);
      if (item.rotation) position.rotation = Number(item.rotation);
      const shape = slide.shapes.add({
        geometry: "rect",
        name: item.name,
        position,
        fill: "#00000000",
        line: { style: "solid", fill: "#00000000", width: 0 },
      });
      shape.text = String(item.text || "");
      shape.text.fontSize = Number(item.fontSize || 14);
      shape.text.typeface = item.typeface || "Microsoft YaHei";
      shape.text.color = normalizeColor(item.color, "#222222");
      shape.text.bold = Boolean(item.bold);
      shape.text.alignment = item.align || "left";
      shape.text.verticalAlignment = item.valign || "middle";
      shape.text.insets = { left: 0, right: 0, top: 0, bottom: 0 };
      shape.text.wrap = "square";
    }
    slideRecords.push({ slide, number: Number(slideSpec.number) });
  }

  await fs.mkdir(path.dirname(out), { recursive: true });
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(out);

  await fs.mkdir(previewDir, { recursive: true });
  await fs.mkdir(layoutDir, { recursive: true });
  const previews = [];
  for (let index = 0; index < slideRecords.length; index += 1) {
    const stem = `slide-${pad(index + 1)}`;
    const previewPath = path.join(previewDir, `${stem}.png`);
    await utils.saveBlobToFile(
      await presentation.export({ slide: slideRecords[index].slide, format: "png", scale: 1 }),
      previewPath,
    );
    await utils.saveBlobToFile(
      await slideRecords[index].slide.export({ format: "layout" }),
      path.join(layoutDir, `${stem}.layout.json`),
    );
    previews.push(previewPath);
  }
  await utils.saveBlobToFile(
    await presentation.export({ format: "webp", montage: true, scale: 1 }),
    path.join(previewDir, "deck-montage.webp"),
  );
  const inspected = await presentation.inspect({
    kind: "slide,textbox,shape,image,layout",
    maxChars: 200000,
  });
  await fs.writeFile(path.join(workspace, "inspect.ndjson"), inspected.ndjson || "", "utf8");
  return { backend: "artifact", previews };
}


function pptxColor(value, fallback = "000000") {
  const normalized = normalizeColor(value, `#${fallback}`).slice(1, 7);
  return normalized || fallback;
}


function inchesPosition(item, slideSpec, slideSize) {
  const pixels = positionFor(item, slideSpec, slideSize);
  const widthInches = 13.333333;
  const heightInches = 7.5;
  return {
    x: (pixels.left / slideSize.width) * widthInches,
    y: (pixels.top / slideSize.height) * heightInches,
    w: (pixels.width / slideSize.width) * widthInches,
    h: (pixels.height / slideSize.height) * heightInches,
  };
}


async function buildWithPptxGen({ manifest, out }) {
  const PptxGenJS = loadPptxGenJS();
  const pptx = new PptxGenJS();
  pptx.author = "photo2slide";
  pptx.subject = "Editable reconstruction of photographed slides";
  pptx.title = "slides";
  pptx.lang = "zh-CN";
  pptx.defineLayout({ name: "PHOTO2SLIDE_WIDE", width: 13.333333, height: 7.5 });
  pptx.layout = "PHOTO2SLIDE_WIDE";
  const slideSize = manifest.slideSize;

  for (const slideSpec of manifest.slides) {
    const slide = pptx.addSlide();
    slide.background = { color: "FFFFFF" };
    slide.addImage({
      path: slideSpec.background,
      x: 0,
      y: 0,
      w: 13.333333,
      h: 7.5,
    });
    for (const item of slideSpec.images || []) {
      slide.addImage({ path: item.file, ...inchesPosition(item, slideSpec, slideSize) });
    }
    for (const item of slideSpec.shapes || []) {
      const transparent = String(item.fill || "").toLowerCase().endsWith("00");
      const transparentLine = String(item.line || "").toLowerCase().endsWith("00");
      slide.addShape(pptx.ShapeType?.rect || "rect", {
        ...inchesPosition(item, slideSpec, slideSize),
        fill: { color: pptxColor(item.fill, "FFFFFF"), transparency: transparent ? 100 : 0 },
        line: {
          color: pptxColor(item.line, "FFFFFF"),
          transparency: transparentLine ? 100 : 0,
          width: Number(item.lineWidth || 0),
        },
        name: item.name,
      });
    }
    for (const item of slideSpec.texts || []) {
      slide.addText(String(item.text || ""), {
        ...inchesPosition(item, slideSpec, slideSize),
        name: item.name,
        margin: 0,
        fontFace: item.typeface || "Microsoft YaHei",
        fontSize: Number(item.fontSizePt || 10.5),
        color: pptxColor(item.color, "222222"),
        bold: Boolean(item.bold),
        align: item.align || "left",
        valign: item.valign === "middle" ? "mid" : item.valign || "mid",
        rotate: Number(item.rotation || 0),
        fit: "shrink",
        fill: { color: "FFFFFF", transparency: 100 },
        line: { color: "FFFFFF", transparency: 100 },
        breakLine: false,
      });
    }
  }
  await fs.mkdir(path.dirname(out), { recursive: true });
  await pptx.writeFile({ fileName: out });
  return { backend: "pptxgenjs", previews: [] };
}


async function main() {
  const args = parseArgs(process.argv.slice(2));
  const manifestPath = requireArg(args, "manifest");
  const out = requireArg(args, "out");
  const workspace = path.resolve(args.workspace || path.join(path.dirname(out), "artifact-workspace"));
  const previewDir = path.resolve(args["preview-dir"] || path.join(workspace, "preview"));
  const layoutDir = path.resolve(args["layout-dir"] || path.join(workspace, "layout"));
  const backend = String(args.backend || "auto");
  if (!["auto", "artifact", "pptxgenjs"].includes(backend)) {
    throw new Error(`Unsupported backend: ${backend}`);
  }
  const manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
  if (!Array.isArray(manifest.slides) || !manifest.slides.length) {
    throw new Error("Manifest does not contain slides");
  }
  await fs.mkdir(workspace, { recursive: true });
  let result;
  const utils = backend === "pptxgenjs" ? null : await findArtifactUtils();
  if (utils) {
    result = await buildWithArtifact({ manifest, out, workspace, previewDir, layoutDir, utils });
  } else {
    if (backend === "artifact") throw new Error("Codex artifact-tool runtime was not found");
    result = await buildWithPptxGen({ manifest, out });
  }
  const stat = await fs.stat(out);
  const buildManifest = {
    output: out,
    bytes: stat.size,
    backend: result.backend,
    slideCount: manifest.slides.length,
    mode: manifest.mode,
    previews: result.previews,
  };
  await fs.writeFile(
    path.join(workspace, "build-manifest.json"),
    `${JSON.stringify(buildManifest, null, 2)}\n`,
    "utf8",
  );
  console.log(JSON.stringify(buildManifest));
}


main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});

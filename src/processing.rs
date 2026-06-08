use std::io::{Cursor, Write};
use std::path::Path;
#[cfg(feature = "tch-backend")]
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use image::codecs::png::PngEncoder;
use image::imageops::FilterType;
use image::{ColorType, DynamicImage, GenericImageView, ImageEncoder, Rgba, RgbaImage};
use rayon::prelude::*;
use zeroize::Zeroize;
use zip::{DateTime, write::SimpleFileOptions};

const ALPHA_THRESHOLD: f32 = 34.0;
const ALPHA_SOFTNESS: f32 = 86.0;

#[derive(Clone, Debug)]
pub enum Background {
    Transparent,
    White,
    Black,
    Image(DynamicImage),
}

#[derive(Debug)]
pub struct InputImage {
    pub filename: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug)]
pub struct ProcessedImage {
    pub filename: String,
    pub png: Vec<u8>,
}

#[allow(dead_code)]
pub trait BackgroundRemover: Send + Sync + 'static {
    fn remove_background(&self, input: &[u8]) -> Result<RgbaImage>;
}

#[derive(Clone, Debug, Default)]
pub struct EdgeColorBackgroundRemover;

impl EdgeColorBackgroundRemover {
    pub fn remove_background_with_thresholds(
        &self,
        input: &[u8],
        threshold: f32,
        softness: f32,
    ) -> Result<RgbaImage> {
        let image = image::load_from_memory(input)
            .context("image invalide")?
            .to_rgba8();
        let bg = estimate_edge_background(&image);
        let mut output = image.clone();

        for pixel in output.pixels_mut() {
            let distance = color_distance(*pixel, bg);
            let alpha = ((distance - threshold) / softness).clamp(0.0, 1.0);
            pixel.0[3] = (alpha * 255.0).round() as u8;
        }

        Ok(output)
    }
}

impl BackgroundRemover for EdgeColorBackgroundRemover {
    fn remove_background(&self, input: &[u8]) -> Result<RgbaImage> {
        self.remove_background_with_thresholds(input, ALPHA_THRESHOLD, ALPHA_SOFTNESS)
    }
}

#[cfg(feature = "tch-backend")]
pub struct TorchScriptBackgroundRemover {
    model: tch::CModule,
    device: tch::Device,
    _python: Option<libloading::Library>,
    _torchvision: Option<libloading::Library>,
}

#[cfg(feature = "tch-backend")]
impl TorchScriptBackgroundRemover {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let device = if tch::Cuda::is_available() {
            tch::Device::Cuda(0)
        } else {
            tch::Device::Cpu
        };
        let python = load_python_library().transpose()?;
        let torchvision = load_torchvision_ops().transpose()?;
        let model = tch::CModule::load_on_device(path, device)
            .context("chargement du modele TorchScript impossible")?;
        Ok(Self {
            model,
            device,
            _python: python,
            _torchvision: torchvision,
        })
    }
}

#[cfg(feature = "tch-backend")]
fn load_torchvision_ops() -> Option<Result<libloading::Library>> {
    let path = find_torchvision_library()?;
    Some(load_library_global(&path).with_context(|| {
        format!(
            "chargement de la bibliotheque torchvision impossible: {}",
            path.display()
        )
    }))
}

#[cfg(feature = "tch-backend")]
fn load_python_library() -> Option<Result<libloading::Library>> {
    let path = find_python_library()?;
    Some(load_library_global(&path).with_context(|| {
        format!(
            "chargement de la bibliotheque Python impossible: {}",
            path.display()
        )
    }))
}

#[cfg(feature = "tch-backend")]
fn load_library_global(path: &Path) -> Result<libloading::Library, libloading::Error> {
    #[cfg(unix)]
    {
        use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
        let library = unsafe { Library::open(Some(path), RTLD_NOW | RTLD_GLOBAL) }?;
        Ok(library.into())
    }

    #[cfg(not(unix))]
    {
        unsafe { libloading::Library::new(path) }
    }
}

#[cfg(feature = "tch-backend")]
fn find_torchvision_library() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("TORCHVISION_LIBRARY_PATH").map(PathBuf::from) {
        return Some(path);
    }

    let candidates = [
        "/usr/local/lib/python3.13/site-packages/torchvision/_C.so",
        "/usr/local/lib/python3.12/site-packages/torchvision/_C.so",
    ];

    candidates
        .iter()
        .map(PathBuf::from)
        .find(|candidate| candidate.exists())
}

#[cfg(feature = "tch-backend")]
fn find_python_library() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("PYTHON_LIBRARY_PATH").map(PathBuf::from) {
        return Some(path);
    }

    let candidates = [
        "/usr/local/lib/libpython3.13.so.1.0",
        "/usr/local/lib/libpython3.12.so.1.0",
    ];

    candidates
        .iter()
        .map(PathBuf::from)
        .find(|candidate| candidate.exists())
}

#[cfg(feature = "tch-backend")]
impl BackgroundRemover for TorchScriptBackgroundRemover {
    fn remove_background(&self, input: &[u8]) -> Result<RgbaImage> {
        use tch::{Kind, Tensor};

        let original = image::load_from_memory(input)
            .context("image invalide")?
            .to_rgb8();
        let mut result = image::load_from_memory(input)
            .context("image invalide")?
            .to_rgba8();
        let (original_width, original_height) = original.dimensions();
        let resized = image::imageops::resize(&original, 1024, 1024, FilterType::Lanczos3);

        let mut normalized = Vec::with_capacity(3 * 1024 * 1024);
        for channel in 0..3 {
            let (mean, std) = match channel {
                0 => (0.485f32, 0.229f32),
                1 => (0.456f32, 0.224f32),
                _ => (0.406f32, 0.225f32),
            };

            for pixel in resized.pixels() {
                normalized.push((pixel.0[channel] as f32 / 255.0 - mean) / std);
            }
        }

        let tensor = Tensor::from_slice(&normalized)
            .reshape([1, 3, 1024, 1024])
            .to_device(self.device);

        let prediction = tch::no_grad(|| self.model.forward_ts(&[tensor]))
            .context("inference TorchScript impossible")?
            .sigmoid()
            .to_device(tch::Device::Cpu)
            .to_kind(Kind::Float)
            .squeeze();

        let mask = Vec::<f32>::try_from(prediction.reshape([1024 * 1024]))
            .context("conversion du masque impossible")?;

        let mut alpha = image::GrayImage::new(1024, 1024);
        for (pixel, value) in alpha.pixels_mut().zip(mask) {
            pixel.0[0] = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        }

        let alpha = image::imageops::resize(
            &alpha,
            original_width,
            original_height,
            FilterType::Lanczos3,
        );
        for (pixel, mask_pixel) in result.pixels_mut().zip(alpha.pixels()) {
            pixel.0[3] = mask_pixel.0[0];
        }

        Ok(result)
    }
}

pub struct ModelRegistry {
    models: Vec<ModelEntry>,
}

pub struct ModelEntry {
    pub id: String,
    pub label: String,
    engine: ModelEngine,
}

pub enum ModelEngine {
    #[cfg(feature = "tch-backend")]
    TorchScript(TorchScriptBackgroundRemover),
}

impl ModelRegistry {
    pub fn load_from_env() -> Result<Self> {
        let mut models = Vec::new();

        for spec in torchscript_model_specs()? {
            models.push(ModelEntry {
                id: spec.id,
                label: spec.label,
                engine: load_torchscript_engine(spec.path)?,
            });
        }

        if models.is_empty() {
            return Err(anyhow!(
                "aucun modele BiRefNet configure: definissez BIREFNET_MODELS ou BIREFNET_TORCHSCRIPT_PATH"
            ));
        }

        Ok(Self { models })
    }

    pub fn options(&self) -> Vec<ModelOption> {
        self.models
            .iter()
            .map(|model| ModelOption {
                id: model.id.clone(),
                label: model.label.clone(),
                is_birefnet: model.engine.is_birefnet(),
            })
            .collect()
    }

    pub fn default_model_id(&self) -> &str {
        self.models
            .iter()
            .find(|model| model.engine.is_birefnet())
            .or_else(|| self.models.first())
            .map(|model| model.id.as_str())
            .unwrap_or("")
    }

    pub fn remove_background(&self, input: &[u8], model_id: &str) -> Result<RgbaImage> {
        let id = if model_id.trim().is_empty() {
            self.default_model_id()
        } else {
            model_id.trim()
        };
        let model = self
            .models
            .iter()
            .find(|model| model.id == id)
            .ok_or_else(|| anyhow!("modele introuvable: {id}"))?;

        match &model.engine {
            #[cfg(feature = "tch-backend")]
            ModelEngine::TorchScript(remover) => remover.remove_background(input),
        }
    }
}

impl ModelEngine {
    fn is_birefnet(&self) -> bool {
        #[cfg(feature = "tch-backend")]
        {
            true
        }

        #[cfg(not(feature = "tch-backend"))]
        {
            false
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelOption {
    pub id: String,
    pub label: String,
    pub is_birefnet: bool,
}

struct TorchScriptModelSpec {
    id: String,
    label: String,
    path: String,
}

fn torchscript_model_specs() -> Result<Vec<TorchScriptModelSpec>> {
    let mut specs = Vec::new();

    if let Ok(value) = std::env::var("BIREFNET_MODELS") {
        for item in value.split(';').map(str::trim).filter(|item| !item.is_empty()) {
            let parts: Vec<_> = item.split('|').map(str::trim).collect();
            match parts.as_slice() {
                [id, label, path] => specs.push(TorchScriptModelSpec {
                    id: sanitize_model_id(id),
                    label: (*label).to_string(),
                    path: (*path).to_string(),
                }),
                [id, path] => specs.push(TorchScriptModelSpec {
                    id: sanitize_model_id(id),
                    label: (*id).to_string(),
                    path: (*path).to_string(),
                }),
                _ => {
                    return Err(anyhow!(
                        "BIREFNET_MODELS invalide: utilisez id|label|path;id2|label2|path2"
                    ));
                }
            }
        }
    }

    if specs.is_empty() {
        if let Ok(path) = std::env::var("BIREFNET_TORCHSCRIPT_PATH") {
            specs.push(TorchScriptModelSpec {
                id: "birefnet-torchscript".to_string(),
                label: "BiRefNet TorchScript".to_string(),
                path,
            });
        }
    }

    Ok(specs)
}

fn sanitize_model_id(value: &str) -> String {
    let id = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string();

    if id.is_empty() {
        "birefnet".to_string()
    } else {
        id
    }
}

#[cfg(feature = "tch-backend")]
fn load_torchscript_engine(path: impl AsRef<Path>) -> Result<ModelEngine> {
    Ok(ModelEngine::TorchScript(TorchScriptBackgroundRemover::load(
        path,
    )?))
}

#[cfg(not(feature = "tch-backend"))]
fn load_torchscript_engine(_path: impl AsRef<Path>) -> Result<ModelEngine> {
    Err(anyhow!(
        "un modele BiRefNet est configure, mais le binaire n'a pas ete compile avec --features tch-backend"
    ))
}

pub fn process_images(
    registry: &ModelRegistry,
    model_id: &str,
    inputs: Vec<InputImage>,
    background: &Background,
) -> Result<Vec<ProcessedImage>> {
    let background = background.clone();

    inputs
        .into_par_iter()
        .map(|mut input| {
            let rgba = registry.remove_background(&input.bytes, model_id)?;
            input.bytes.zeroize();
            let composed = apply_background(rgba, &background);
            let png = encode_png(&composed)?;
            Ok(ProcessedImage {
                filename: result_filename(&input.filename),
                png,
            })
        })
        .collect()
}

pub fn apply_background(foreground: RgbaImage, background: &Background) -> RgbaImage {
    match background {
        Background::Transparent => foreground,
        Background::White => composite_on_solid(foreground, Rgba([255, 255, 255, 255])),
        Background::Black => composite_on_solid(foreground, Rgba([0, 0, 0, 255])),
        Background::Image(bg) => composite_on_image(foreground, bg),
    }
}

pub fn encode_png(image: &RgbaImage) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let encoder = PngEncoder::new(&mut bytes);
    encoder
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            ColorType::Rgba8.into(),
        )
        .context("encodage PNG impossible")?;
    Ok(bytes)
}

pub fn zip_images(images: &[ProcessedImage]) -> Result<Vec<u8>> {
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut cursor);
        let options = SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .last_modified_time(DateTime::default());

        for image in images {
            zip.start_file(&image.filename, options)
                .context("creation entree ZIP impossible")?;
            zip.write_all(&image.png)
                .context("ecriture entree ZIP impossible")?;
        }

        zip.finish().context("finalisation ZIP impossible")?;
    }
    Ok(cursor.into_inner())
}

pub fn decode_background_image(bytes: &[u8]) -> Result<DynamicImage> {
    image::load_from_memory(bytes).context("image de fond invalide")
}

pub fn sanitize_filename(name: &str) -> String {
    let basename = name
        .rsplit(['/', '\\'])
        .next()
        .filter(|part| !part.is_empty())
        .unwrap_or("image.png");

    let cleaned: String = basename
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect();

    let trimmed = cleaned.trim_matches('_');
    if trimmed.is_empty() {
        "image.png".to_string()
    } else {
        trimmed.to_string()
    }
}

fn result_filename(original: &str) -> String {
    let sanitized = sanitize_filename(original);
    let stem = sanitized.rsplit_once('.').map_or(sanitized.as_str(), |(stem, _)| stem);
    format!("{stem}-birefnet.png")
}

fn composite_on_solid(foreground: RgbaImage, color: Rgba<u8>) -> RgbaImage {
    let mut background = RgbaImage::from_pixel(foreground.width(), foreground.height(), color);
    image::imageops::overlay(&mut background, &foreground, 0, 0);
    force_opaque(&mut background);
    background
}

fn composite_on_image(foreground: RgbaImage, background: &DynamicImage) -> RgbaImage {
    let (width, height) = foreground.dimensions();
    let mut fitted = contain_resize(background, width, height).to_rgba8();
    image::imageops::overlay(&mut fitted, &foreground, 0, 0);
    force_opaque(&mut fitted);
    fitted
}

fn force_opaque(image: &mut RgbaImage) {
    for pixel in image.pixels_mut() {
        pixel.0[3] = 255;
    }
}

fn contain_resize(image: &DynamicImage, width: u32, height: u32) -> DynamicImage {
    let (source_width, source_height) = image.dimensions();
    if source_width == 0 || source_height == 0 || width == 0 || height == 0 {
        return image.resize_exact(width.max(1), height.max(1), FilterType::Lanczos3);
    }

    let source_ratio = source_width as f32 / source_height as f32;
    let target_ratio = width as f32 / height as f32;

    let resized = if source_ratio > target_ratio {
        image.resize(width, (width as f32 / source_ratio).round() as u32, FilterType::Lanczos3)
    } else {
        image.resize(
            (height as f32 * source_ratio).round() as u32,
            height,
            FilterType::Lanczos3,
        )
    };

    let resized = resized.to_rgba8();
    let mut canvas = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 255]));
    let x = width.saturating_sub(resized.width()) / 2;
    let y = height.saturating_sub(resized.height()) / 2;
    image::imageops::overlay(&mut canvas, &resized, x.into(), y.into());
    DynamicImage::ImageRgba8(canvas)
}

fn estimate_edge_background(image: &RgbaImage) -> Rgba<u8> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Rgba([255, 255, 255, 255]);
    }

    let mut count = 0u64;
    let mut sum = [0u64; 3];

    for x in 0..width {
        accumulate_rgb(image.get_pixel(x, 0), &mut sum, &mut count);
        accumulate_rgb(image.get_pixel(x, height - 1), &mut sum, &mut count);
    }

    for y in 0..height {
        accumulate_rgb(image.get_pixel(0, y), &mut sum, &mut count);
        accumulate_rgb(image.get_pixel(width - 1, y), &mut sum, &mut count);
    }

    if count == 0 {
        return Rgba([255, 255, 255, 255]);
    }

    Rgba([
        (sum[0] / count) as u8,
        (sum[1] / count) as u8,
        (sum[2] / count) as u8,
        255,
    ])
}

fn accumulate_rgb(pixel: &Rgba<u8>, sum: &mut [u64; 3], count: &mut u64) {
    sum[0] += pixel.0[0] as u64;
    sum[1] += pixel.0[1] as u64;
    sum[2] += pixel.0[2] as u64;
    *count += 1;
}

fn color_distance(a: Rgba<u8>, b: Rgba<u8>) -> f32 {
    let dr = a.0[0] as f32 - b.0[0] as f32;
    let dg = a.0[1] as f32 - b.0[1] as f32;
    let db = a.0[2] as f32 - b.0[2] as f32;
    (dr * dr + dg * dg + db * db).sqrt()
}

pub fn require_images(images: Vec<InputImage>) -> Result<Vec<InputImage>> {
    if images.is_empty() {
        Err(anyhow!("aucune image recue"))
    } else {
        Ok(images)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_filename_removes_path_and_shell_characters() {
        assert_eq!(sanitize_filename("../photo final.png"), "photo_final.png");
        assert_eq!(sanitize_filename("portrait;$HOME.jpg"), "portrait__HOME.jpg");
        assert_eq!(sanitize_filename("###"), "image.png");
    }

    #[test]
    fn zip_images_writes_all_outputs() {
        let images = vec![
            ProcessedImage {
                filename: "one.png".to_string(),
                png: vec![1, 2, 3],
            },
            ProcessedImage {
                filename: "two.png".to_string(),
                png: vec![4, 5, 6],
            },
        ];

        let bytes = zip_images(&images).expect("zip");
        let reader = Cursor::new(bytes);
        let mut zip = zip::ZipArchive::new(reader).expect("archive");

        assert_eq!(zip.len(), 2);
        for index in 0..zip.len() {
            let file = zip.by_index(index).expect("entry");
            assert_eq!(
                file.last_modified().expect("mtime").to_string(),
                "1980-01-01 00:00:00"
            );
        }
    }

    #[test]
    fn solid_background_removes_transparency() {
        let mut image = RgbaImage::new(1, 1);
        image.put_pixel(0, 0, Rgba([255, 0, 0, 128]));

        let output = apply_background(image, &Background::White);
        assert_eq!(output.get_pixel(0, 0).0[3], 255);
    }

    #[test]
    fn image_background_is_contained_without_cropping() {
        let foreground = RgbaImage::from_pixel(100, 100, Rgba([255, 0, 0, 0]));
        let mut background = RgbaImage::from_pixel(200, 50, Rgba([0, 0, 255, 255]));
        for y in 0..50 {
            for x in 0..20 {
                background.put_pixel(x, y, Rgba([255, 255, 0, 255]));
                background.put_pixel(199 - x, y, Rgba([0, 255, 0, 255]));
            }
        }

        let output = apply_background(
            foreground,
            &Background::Image(DynamicImage::ImageRgba8(background)),
        );

        assert_eq!(output.dimensions(), (100, 100));
        assert!(output.get_pixel(2, 50).0[0] > 200);
        assert!(output.get_pixel(97, 50).0[1] > 200);
    }
}

"""
FinCo AI - Image Parser

Provides image ingestion and OCR support for:
- PNG
- JPG / JPEG
- GIF
- BMP
- TIFF
- WEBP

Capabilities:
- Image metadata extraction
- Image validation
- OCR text extraction using Tesseract
- Optional preprocessing for better OCR
- Page/document image support
- Text quality reporting
- Safe image size limits
- Conversion to PIL Image
- Structured parsing results

Required dependencies:
    Pillow
    pytesseract

Optional:
    OpenCV is not required. Pillow is used for preprocessing.

System dependency:
    Tesseract OCR must be installed separately.

Ubuntu/Debian:
    sudo apt-get install tesseract-ocr

Windows:
    Install Tesseract OCR and configure the executable path.

Example:

    from app.ingestion.image_parser import image_parser

    result = image_parser.parse(
        file_path="data/raw/invoice.png"
    )

    print(result.ocr_text)
    print(result.metadata)
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
    ImageOps,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class ImageParserError(Exception):
    """Base exception for image parser errors."""


class ImageFileNotFoundError(ImageParserError):
    """Raised when the image file does not exist."""


class UnsupportedImageFormatError(ImageParserError):
    """Raised when the image format is unsupported."""


class ImageValidationError(ImageParserError):
    """Raised when the image fails validation."""


class OCRNotAvailableError(ImageParserError):
    """Raised when pytesseract or Tesseract is unavailable."""


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class ImageParserConfig:
    """
    Configuration for image parsing and OCR.
    """

    supported_extensions: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    )

    supported_formats: tuple[str, ...] = (
        "PNG",
        "JPEG",
        "GIF",
        "BMP",
        "TIFF",
        "WEBP",
    )

    max_file_size_bytes: int = (
        25 * 1024 * 1024
    )

    max_pixels: int = (
        50_000_000
    )

    enable_ocr: bool = True

    preprocess_for_ocr: bool = True

    grayscale_for_ocr: bool = True

    increase_contrast: bool = True

    sharpen_for_ocr: bool = True

    threshold_for_ocr: bool = False

    ocr_language: str = "eng"

    ocr_config: str = "--psm 6"

    tesseract_cmd: Optional[str] = None

    preserve_original_orientation: bool = True

    extract_exif_metadata: bool = True


# ============================================================================
# Result Models
# ============================================================================


@dataclass
class ImageQualityReport:
    """
    Quality and validation report for an image.
    """

    width: int

    height: int

    format: Optional[str]

    mode: str

    file_size_bytes: int

    has_alpha: bool

    is_animated: bool

    frame_count: int

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    is_valid: bool = True

    def add_warning(
        self,
        message: str,
    ) -> None:
        self.warnings.append(message)

    def add_error(
        self,
        message: str,
    ) -> None:
        self.errors.append(message)
        self.is_valid = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "mode": self.mode,
            "file_size_bytes": self.file_size_bytes,
            "has_alpha": self.has_alpha,
            "is_animated": self.is_animated,
            "frame_count": self.frame_count,
            "warnings": self.warnings,
            "errors": self.errors,
            "is_valid": self.is_valid,
        }


@dataclass
class OCRResult:
    """
    OCR result for an image.
    """

    text: str

    confidence: Optional[float]

    language: str

    word_count: int

    line_count: int

    character_count: int

    success: bool = True

    warnings: List[str] = field(
        default_factory=list
    )

    raw_data: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "word_count": self.word_count,
            "line_count": self.line_count,
            "character_count": self.character_count,
            "success": self.success,
            "warnings": self.warnings,
        }


@dataclass
class ImageParseResult:
    """
    Final image parsing result.
    """

    file_path: str

    file_name: str

    file_extension: str

    mime_type: Optional[str]

    image_format: Optional[str]

    width: int

    height: int

    mode: str

    ocr_text: str

    ocr_result: Optional[OCRResult]

    quality_report: ImageQualityReport

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    success: bool = True

    @property
    def has_text(self) -> bool:
        return bool(
            self.ocr_text.strip()
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_extension": self.file_extension,
            "mime_type": self.mime_type,
            "image_format": self.image_format,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "ocr_text": self.ocr_text,
            "ocr_result": (
                self.ocr_result.to_dict()
                if self.ocr_result
                else None
            ),
            "quality_report": (
                self.quality_report.to_dict()
            ),
            "metadata": self.metadata,
            "warnings": self.warnings,
            "errors": self.errors,
            "success": self.success,
        }


# ============================================================================
# Image Parser
# ============================================================================


class ImageParser:
    """
    Parses image files and optionally extracts text through OCR.
    """

    def __init__(
        self,
        config: ImageParserConfig = ImageParserConfig(),
    ) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        file_path: str | Path,
        source_name: Optional[str] = None,
        run_ocr: Optional[bool] = None,
    ) -> ImageParseResult:
        """
        Parse an image file.

        Args:
            file_path:
                Path to the image.

            source_name:
                Optional logical source name.

            run_ocr:
                Overrides config.enable_ocr when provided.

        Returns:
            ImageParseResult
        """

        path = Path(file_path)

        self._validate_file_path(path)

        image = self._open_image(path)

        quality_report = (
            self._build_quality_report(
                path=path,
                image=image,
            )
        )

        warnings = list(
            quality_report.warnings
        )

        errors = list(
            quality_report.errors
        )

        ocr_result: Optional[OCRResult] = None
        ocr_text = ""

        should_run_ocr = (
            self.config.enable_ocr
            if run_ocr is None
            else run_ocr
        )

        if (
            should_run_ocr
            and quality_report.is_valid
        ):
            try:
                ocr_result = self.extract_text(
                    image=image
                )

                ocr_text = ocr_result.text

                warnings.extend(
                    ocr_result.warnings
                )

            except OCRNotAvailableError as exc:
                message = str(exc)

                logger.warning(message)
                warnings.append(message)

            except Exception as exc:
                message = (
                    f"OCR failed for '{path.name}': "
                    f"{exc}"
                )

                logger.exception(message)
                warnings.append(message)

        metadata = self._extract_metadata(
            image=image,
            path=path,
            source_name=source_name,
        )

        return ImageParseResult(
            file_path=str(path),
            file_name=path.name,
            file_extension=path.suffix.lower(),
            mime_type=self._guess_mime_type(
                path.suffix.lower()
            ),
            image_format=image.format,
            width=image.width,
            height=image.height,
            mode=image.mode,
            ocr_text=ocr_text,
            ocr_result=ocr_result,
            quality_report=quality_report,
            metadata=metadata,
            warnings=warnings,
            errors=errors,
            success=quality_report.is_valid,
        )

    def parse_bytes(
        self,
        content: bytes,
        file_name: str = "uploaded_image",
        source_name: Optional[str] = None,
        run_ocr: Optional[bool] = None,
    ) -> ImageParseResult:
        """
        Parse image bytes without requiring a filesystem path.
        """

        if not content:
            raise ImageValidationError(
                "Image content is empty."
            )

        if (
            len(content)
            > self.config.max_file_size_bytes
        ):
            raise ImageValidationError(
                "Image exceeds the configured size limit."
            )

        try:
            image = Image.open(
                io.BytesIO(content)
            )

            image.load()

        except Exception as exc:
            raise ImageValidationError(
                f"Unable to decode image bytes: {exc}"
            ) from exc

        quality_report = (
            self._build_quality_report_from_image(
                image=image,
                file_size_bytes=len(content),
            )
        )

        warnings = list(
            quality_report.warnings
        )

        errors = list(
            quality_report.errors
        )

        ocr_result: Optional[OCRResult] = None
        ocr_text = ""

        should_run_ocr = (
            self.config.enable_ocr
            if run_ocr is None
            else run_ocr
        )

        if (
            should_run_ocr
            and quality_report.is_valid
        ):
            try:
                ocr_result = self.extract_text(
                    image=image
                )

                ocr_text = ocr_result.text
                warnings.extend(
                    ocr_result.warnings
                )

            except OCRNotAvailableError as exc:
                warnings.append(str(exc))

            except Exception as exc:
                warnings.append(
                    f"OCR failed: {exc}"
                )

        metadata = self._extract_metadata(
            image=image,
            path=None,
            source_name=source_name,
        )

        return ImageParseResult(
            file_path="",
            file_name=file_name,
            file_extension=Path(
                file_name
            ).suffix.lower(),
            mime_type=self._guess_mime_type(
                Path(file_name).suffix.lower()
            ),
            image_format=image.format,
            width=image.width,
            height=image.height,
            mode=image.mode,
            ocr_text=ocr_text,
            ocr_result=ocr_result,
            quality_report=quality_report,
            metadata=metadata,
            warnings=warnings,
            errors=errors,
            success=quality_report.is_valid,
        )

    def extract_text(
        self,
        image: Image.Image,
    ) -> OCRResult:
        """
        Extract text from an image using Tesseract OCR.
        """

        pytesseract = self._load_pytesseract()

        working_image = image.copy()

        if self.config.preserve_original_orientation:
            working_image = ImageOps.exif_transpose(
                working_image
            )

        if self.config.preprocess_for_ocr:
            working_image = (
                self._preprocess_for_ocr(
                    working_image
                )
            )

        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = (
                self.config.tesseract_cmd
            )

        text = pytesseract.image_to_string(
            working_image,
            lang=self.config.ocr_language,
            config=self.config.ocr_config,
        )

        cleaned_text = self._clean_ocr_text(
            text
        )

        confidence = (
            self._calculate_ocr_confidence(
                image=working_image,
                pytesseract=pytesseract,
            )
        )

        raw_data = self._extract_ocr_data(
            image=working_image,
            pytesseract=pytesseract,
        )

        words = self._split_words(
            cleaned_text
        )

        lines = [
            line
            for line in cleaned_text.splitlines()
            if line.strip()
        ]

        return OCRResult(
            text=cleaned_text,
            confidence=confidence,
            language=self.config.ocr_language,
            word_count=len(words),
            line_count=len(lines),
            character_count=len(cleaned_text),
            success=True,
            warnings=[],
            raw_data=raw_data,
        )

    def validate(
        self,
        file_path: str | Path,
    ) -> ImageQualityReport:
        """
        Validate an image without running OCR.
        """

        path = Path(file_path)

        self._validate_file_path(path)

        image = self._open_image(path)

        return self._build_quality_report(
            path=path,
            image=image,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_file_path(
        self,
        path: Path,
    ) -> None:
        if not path.exists():
            raise ImageFileNotFoundError(
                f"Image file not found: {path}"
            )

        if not path.is_file():
            raise ImageValidationError(
                f"Image path is not a file: {path}"
            )

        if path.suffix.lower() not in (
            self.config.supported_extensions
        ):
            raise UnsupportedImageFormatError(
                f"Unsupported image extension: "
                f"{path.suffix}"
            )

        file_size = path.stat().st_size

        if (
            file_size
            > self.config.max_file_size_bytes
        ):
            raise ImageValidationError(
                "Image exceeds maximum allowed size "
                f"of {self.config.max_file_size_bytes} bytes."
            )

    def _open_image(
        self,
        path: Path,
    ) -> Image.Image:
        try:
            image = Image.open(path)

            image.verify()

            image = Image.open(path)

            image.load()

            if image.format not in (
                self.config.supported_formats
            ):
                raise UnsupportedImageFormatError(
                    f"Unsupported decoded image format: "
                    f"{image.format}"
                )

            return image

        except UnsupportedImageFormatError:
            raise

        except Exception as exc:
            raise ImageValidationError(
                f"Unable to open image '{path}': {exc}"
            ) from exc

    def _build_quality_report(
        self,
        path: Path,
        image: Image.Image,
    ) -> ImageQualityReport:
        return self._build_quality_report_from_image(
            image=image,
            file_size_bytes=path.stat().st_size,
        )

    def _build_quality_report_from_image(
        self,
        image: Image.Image,
        file_size_bytes: int,
    ) -> ImageQualityReport:
        frame_count = self._get_frame_count(
            image
        )

        is_animated = (
            frame_count > 1
        )

        has_alpha = (
            "A" in image.getbands()
            or image.mode in {
                "LA",
                "PA",
            }
        )

        report = ImageQualityReport(
            width=image.width,
            height=image.height,
            format=image.format,
            mode=image.mode,
            file_size_bytes=file_size_bytes,
            has_alpha=has_alpha,
            is_animated=is_animated,
            frame_count=frame_count,
        )

        total_pixels = (
            image.width * image.height
        )

        if (
            total_pixels
            > self.config.max_pixels
        ):
            report.add_error(
                "Image exceeds the maximum allowed "
                f"pixel count of {self.config.max_pixels}."
            )

        if image.width < 100 or image.height < 100:
            report.add_warning(
                "Image dimensions are very small. "
                "OCR quality may be poor."
            )

        if image.width < 800:
            report.add_warning(
                "Image width is below 800 pixels. "
                "Small text may not be detected reliably."
            )

        if is_animated:
            report.add_warning(
                "Animated image detected. "
                "Only the current frame will be processed."
            )

        if image.mode == "P":
            report.add_warning(
                "Palette-based image detected. "
                "It may be converted during OCR preprocessing."
            )

        return report

    # ------------------------------------------------------------------
    # OCR preprocessing
    # ------------------------------------------------------------------

    def _preprocess_for_ocr(
        self,
        image: Image.Image,
    ) -> Image.Image:
        processed = image

        if self.config.grayscale_for_ocr:
            processed = ImageOps.grayscale(
                processed
            )

        if self.config.increase_contrast:
            enhancer = ImageEnhance.Contrast(
                processed
            )

            processed = enhancer.enhance(
                1.8
            )

        if self.config.sharpen_for_ocr:
            processed = processed.filter(
                ImageFilter.SHARPEN
            )

        if self.config.threshold_for_ocr:
            processed = processed.point(
                lambda pixel: (
                    255
                    if pixel > 160
                    else 0
                )
            )

        return processed

    # ------------------------------------------------------------------
    # OCR helpers
    # ------------------------------------------------------------------

    def _load_pytesseract(self):
        try:
            import pytesseract

            return pytesseract

        except ImportError as exc:
            raise OCRNotAvailableError(
                "pytesseract is not installed. "
                "Install it with: pip install pytesseract"
            ) from exc

    def _calculate_ocr_confidence(
        self,
        image: Image.Image,
        pytesseract,
    ) -> Optional[float]:
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.config.ocr_language,
                config=self.config.ocr_config,
                output_type=(
                    pytesseract.Output.DICT
                ),
            )

            confidence_values = []

            for value in data.get(
                "conf",
                [],
            ):
                try:
                    confidence = float(value)

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if confidence >= 0:
                    confidence_values.append(
                        confidence
                    )

            if not confidence_values:
                return None

            return round(
                sum(confidence_values)
                / len(confidence_values),
                2,
            )

        except Exception as exc:
            logger.debug(
                "Unable to calculate OCR confidence: %s",
                exc,
            )

            return None

    def _extract_ocr_data(
        self,
        image: Image.Image,
        pytesseract,
    ) -> Dict[str, Any]:
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.config.ocr_language,
                config=self.config.ocr_config,
                output_type=(
                    pytesseract.Output.DICT
                ),
            )

            return {
                "text": data.get(
                    "text",
                    [],
                ),
                "confidence": data.get(
                    "conf",
                    [],
                ),
                "left": data.get(
                    "left",
                    [],
                ),
                "top": data.get(
                    "top",
                    [],
                ),
                "width": data.get(
                    "width",
                    [],
                ),
                "height": data.get(
                    "height",
                    [],
                ),
            }

        except Exception as exc:
            logger.debug(
                "Unable to extract raw OCR data: %s",
                exc,
            )

            return {}

    def _clean_ocr_text(
        self,
        text: str,
    ) -> str:
        if not text:
            return ""

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    def _split_words(
        self,
        text: str,
    ) -> List[str]:
        return re.findall(
            r"\S+",
            text,
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _extract_metadata(
        self,
        image: Image.Image,
        path: Optional[Path],
        source_name: Optional[str],
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "source_name": source_name,
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "has_transparency": (
                "transparency" in image.info
            ),
            "dpi": image.info.get(
                "dpi"
            ),
        }

        if self.config.extract_exif_metadata:
            try:
                exif = image.getexif()

                if exif:
                    metadata["exif"] = {
                        str(key): self._safe_metadata_value(
                            value
                        )
                        for key, value in exif.items()
                    }

            except Exception as exc:
                logger.debug(
                    "Unable to extract EXIF metadata: %s",
                    exc,
                )

        if path is not None:
            metadata.update(
                {
                    "file_name": path.name,
                    "file_extension": path.suffix.lower(),
                    "file_size_bytes": path.stat().st_size,
                }
            )

        return metadata

    def _safe_metadata_value(
        self,
        value: Any,
    ) -> Any:
        if isinstance(
            value,
            bytes,
        ):
            return value.hex()

        if isinstance(
            value,
            (str, int, float, bool),
        ) or value is None:
            return value

        return str(value)

    def _get_frame_count(
        self,
        image: Image.Image,
    ) -> int:
        try:
            return int(
                getattr(
                    image,
                    "n_frames",
                    1,
                )
            )

        except Exception:
            return 1

    def _guess_mime_type(
        self,
        extension: str,
    ) -> Optional[str]:
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
            ".webp": "image/webp",
        }

        return mime_types.get(
            extension
        )


# ============================================================================
# Singleton
# ============================================================================


image_parser = ImageParser()
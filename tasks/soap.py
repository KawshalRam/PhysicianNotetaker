from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SOAPNote:
    """Data class for structured SOAP note"""
    Subjective: Dict[str, str]
    Objective: Dict[str, str]
    Assessment: Dict[str, str]
    Plan: Dict[str, str]
    metadata: Optional[Dict] = None


class SOAPNoteGenerator:
    """
    Production-grade SOAP Note Generator with robust error handling,
    long text processing, and optimized inference.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "stabilityai/stablelm-2-zephyr-1_6b",
        use_4bit: bool = False,
        max_memory_gb: Optional[float] = None
    ):
        """
        Initialize the SOAP Note Generator.
        
        Args:
            model_name_or_path: HuggingFace model identifier
            use_4bit: Enable 4-bit quantization for memory efficiency
            max_memory_gb: Maximum GPU memory to use (None = unlimited)
        """
        logger.info(f"Loading model: {model_name_or_path}")
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        if self.device == "cpu":
            logger.warning("⚠️ Running on CPU. This will be slow. GPU recommended.")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Get model configuration
        self.max_context_length = self._get_model_max_length()
        logger.info(f"Model max context length: {self.max_context_length} tokens")
        
        # Load model with optional quantization
        model_kwargs = {
            "device_map": "auto" if self.device == "cuda" else None,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
        
        if self.device == "cuda":
            if use_4bit:
                # 4-bit quantization for memory efficiency
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("Using 4-bit quantization")
            else:
                model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Calculate and log memory usage
        self._log_model_info()
        
    def _get_model_max_length(self) -> int:
        """Get model's maximum context length"""
        # StableLM-2-Zephyr models typically support 4096 tokens
        if hasattr(self.tokenizer, 'model_max_length'):
            max_len = self.tokenizer.model_max_length
            # Some tokenizers have unrealistic defaults
            if max_len > 1_000_000:
                return 4096
            return max_len
        return 4096
    
    def _log_model_info(self):
        """Log model size and memory information"""
        params = sum(p.numel() for p in self.model.parameters())
        params_billion = params / 1e9
        
        # Estimate memory (2 bytes per param for FP16, 4 for FP32)
        dtype_size = 2 if self.model.dtype == torch.float16 else 4
        memory_gb = (params * dtype_size) / (1024**3)
        
        logger.info(f"Model parameters: {params_billion:.2f}B")
        logger.info(f"Estimated memory: {memory_gb:.2f} GB")
    
    def preprocess_transcript(self, transcript: str) -> str:
        """
        Clean and preprocess medical transcript.
        
        Args:
            transcript: Raw conversation text
            
        Returns:
            Cleaned transcript
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', transcript)
        
        # Remove special characters that might confuse the model
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize speaker labels
        text = re.sub(r'(?i)(doctor|dr\.?|physician):', 'Doctor:', text)
        text = re.sub(r'(?i)(patient|pt\.?):', 'Patient:', text)
        
        return text.strip()
    
    def chunk_text(
        self,
        text: str,
        max_chunk_tokens: int = 1024,
        overlap_tokens: int = 100
    ) -> List[str]:
        """
        Split long text into overlapping chunks to handle context limits.
        
        Args:
            text: Input text to chunk
            max_chunk_tokens: Maximum tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
            
        Returns:
            List of text chunks
        """
        # Tokenize the full text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # If text fits in one chunk, return as-is
        if len(tokens) <= max_chunk_tokens:
            return [text]
        
        logger.warning(
            f"Text exceeds {max_chunk_tokens} tokens ({len(tokens)} total). "
            f"Splitting into chunks with {overlap_tokens} token overlap."
        )
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Get chunk tokens
            end_idx = min(start_idx + max_chunk_tokens, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            
            # Move to next chunk with overlap
            if end_idx >= len(tokens):
                break
            start_idx = end_idx - overlap_tokens
        
        logger.info(f"Created {len(chunks)} chunks from long text")
        return chunks
    
    def create_prompt(self, transcript: str) -> str:
        """
        Create optimized prompt using tokenizer's chat template.
        
        Args:
            transcript: Preprocessed conversation transcript
            
        Returns:
            Formatted prompt string
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical documentation assistant. Extract information "
                    "from doctor-patient conversations and create SOAP notes. "
                    "Output ONLY valid JSON with no additional text."
                )
            },
            {
                "role": "user",
                "content": f"""Conversation:
{transcript}

Create a SOAP note in this EXACT JSON format (use ONLY this structure):
{{
  "Subjective": {{
    "Chief_Complaint": "main patient complaint",
    "History_of_Present_Illness": "symptom history and details"
  }},
  "Objective": {{
    "Physical_Exam": "exam findings or NOT DOCUMENTED",
    "Observations": "clinical observations or NOT DOCUMENTED"
  }},
  "Assessment": {{
    "Diagnosis": "clinical diagnosis or PENDING EVALUATION",
    "Severity": "condition severity or UNKNOWN"
  }},
  "Plan": {{
    "Treatment": "treatment recommendations",
    "Follow_Up": "follow-up plan"
  }}
}}

Rules:
- Output ONLY the JSON object
- Use "NOT DOCUMENTED" if information is missing
- Be concise and clinical
- Include only information from the conversation"""
            }
        ]
        
        # Use tokenizer's built-in chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def estimate_prompt_tokens(self, prompt: str) -> int:
        """Estimate number of tokens in prompt"""
        return len(self.tokenizer.encode(prompt, add_special_tokens=True))
    
    def generate_soap_note(
        self,
        transcript: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        chunk_long_text: bool = True
    ) -> Dict:
        """
        Generate SOAP note from transcript with robust error handling.
        
        Args:
            transcript: Doctor-patient conversation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            chunk_long_text: Whether to chunk text if too long
            
        Returns:
            Dictionary containing SOAP note
        """
        try:
            # Preprocess transcript
            clean_transcript = self.preprocess_transcript(transcript)
            
            # Handle long transcripts
            if chunk_long_text:
                # Reserve space for prompt template and generation
                available_tokens = self.max_context_length - max_new_tokens - 500
                chunks = self.chunk_text(clean_transcript, max_chunk_tokens=available_tokens)
                
                if len(chunks) > 1:
                    logger.info(f"Processing {len(chunks)} chunks separately")
                    # Process first chunk (usually contains most relevant info)
                    clean_transcript = chunks[0]
                    logger.warning(
                        "⚠️ Using only first chunk. Consider using a model with "
                        "larger context or implement chunk merging."
                    )
            
            # Create prompt
            prompt = self.create_prompt(clean_transcript)
            
            # Check prompt length
            prompt_tokens = self.estimate_prompt_tokens(prompt)
            total_tokens = prompt_tokens + max_new_tokens
            
            if total_tokens > self.max_context_length:
                logger.error(
                    f"Prompt ({prompt_tokens}) + generation ({max_new_tokens}) "
                    f"exceeds max length ({self.max_context_length})"
                )
                raise ValueError(
                    f"Input too long: {prompt_tokens} tokens. "
                    f"Max allowed: {self.max_context_length - max_new_tokens}"
                )
            
            logger.info(
                f"Prompt: {prompt_tokens} tokens | "
                f"Generation: {max_new_tokens} tokens | "
                f"Total: {total_tokens}/{self.max_context_length}"
            )
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_length - max_new_tokens
            ).to(self.model.device)
            
            # Generate with optimized parameters
            logger.info("Generating SOAP note...")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    top_k=50,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True  # Enable KV cache for efficiency
                )
            
            # Decode only the generated portion
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            
            logger.info(f"Generated {len(generated_tokens)} tokens")
            
            # Extract and parse JSON
            soap_json = self.extract_json(generated_text)
            
            if soap_json is None:
                logger.warning("Failed to parse JSON from model output")
                logger.debug(f"Raw output: {generated_text[:500]}...")
                soap_json = self.create_fallback_soap(clean_transcript, generated_text)
            else:
                logger.info("✓ Successfully generated SOAP note")
            
            return soap_json
            
        except Exception as e:
            logger.error(f"Error generating SOAP note: {str(e)}")
            return self.create_fallback_soap(transcript, error=str(e))
    
    def extract_json(self, generated_text: str) -> Optional[Dict]:
        """
        Extract and validate JSON from generated text.
        Multiple extraction strategies for robustness.
        
        Args:
            generated_text: Text generated by model
            
        Returns:
            Parsed JSON dict or None
        """
        # Strategy 1: Extract from code blocks
        code_block_pattern = r'``````'
        matches = re.findall(code_block_pattern, generated_text, re.DOTALL)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                if self._validate_soap_structure(parsed):
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # Strategy 2: Find first complete JSON object
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        matches = re.findall(json_pattern, generated_text, re.DOTALL)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                if self._validate_soap_structure(parsed):
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # Strategy 3: Try to fix common JSON errors
        try:
            # Remove trailing commas
            fixed_text = re.sub(r',(\s*[}\]])', r'\1', generated_text)
            # Extract JSON-like content
            start = fixed_text.find('{')
            end = fixed_text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = fixed_text[start:end]
                parsed = json.loads(json_str)
                if self._validate_soap_structure(parsed):
                    return parsed
        except:
            pass
        
        return None
    
    def _validate_soap_structure(self, parsed_json: Dict) -> bool:
        """
        Validate that JSON has required SOAP sections.
        
        Args:
            parsed_json: Parsed JSON dictionary
            
        Returns:
            True if valid SOAP structure
        """
        required_sections = ["Subjective", "Objective", "Assessment", "Plan"]
        
        # Check all main sections exist
        if not all(section in parsed_json for section in required_sections):
            return False
        
        # Check that sections are dictionaries
        for section in required_sections:
            if not isinstance(parsed_json[section], dict):
                return False
            # Each section should have at least one field
            if len(parsed_json[section]) == 0:
                return False
        
        return True
    
    def create_fallback_soap(
        self,
        transcript: str,
        generated_text: str = "",
        error: str = ""
    ) -> Dict:
        """
        Create fallback SOAP note when parsing fails.
        
        Args:
            transcript: Original transcript
            generated_text: Text that failed to parse
            error: Error message if applicable
            
        Returns:
            Basic SOAP note structure
        """
        fallback = {
            "Subjective": {
                "Chief_Complaint": "EXTRACTION FAILED - Manual review required",
                "History_of_Present_Illness": "See original transcript"
            },
            "Objective": {
                "Physical_Exam": "NOT DOCUMENTED",
                "Observations": "NOT DOCUMENTED"
            },
            "Assessment": {
                "Diagnosis": "PENDING CLINICAL EVALUATION",
                "Severity": "UNKNOWN"
            },
            "Plan": {
                "Treatment": "TO BE DETERMINED",
                "Follow_Up": "TO BE SCHEDULED"
            },
            "_metadata": {
                "status": "FALLBACK",
                "requires_review": True,
                "original_transcript_preview": transcript[:500],
                "generated_text_preview": generated_text[:500] if generated_text else None,
                "error": error if error else "JSON parsing failed"
            }
        }
        
        return fallback
    
    def format_soap_note(self, soap_dict: Dict) -> str:
        """
        Format SOAP note dictionary into readable text.
        
        Args:
            soap_dict: SOAP note dictionary
            
        Returns:
            Formatted string
        """
        formatted = "=" * 70 + "\n"
        formatted += "SOAP NOTE\n"
        formatted += "=" * 70 + "\n\n"
        
        # Check if this is a fallback note
        if "_metadata" in soap_dict and soap_dict["_metadata"].get("status") == "FALLBACK":
            formatted += "⚠️ WARNING: This is a fallback note. Manual review required.\n\n"
        
        for section, content in soap_dict.items():
            if section.startswith("_"):
                continue
                
            formatted += f"{section.upper()}\n"
            formatted += "-" * 70 + "\n"
            
            if isinstance(content, dict):
                for key, value in content.items():
                    formatted += f"  {key.replace('_', ' ')}: {value}\n"
            else:
                formatted += f"  {content}\n"
            
            formatted += "\n"
        
        return formatted
    
    def save_soap_note(
        self,
        soap_dict: Dict,
        output_path: str = "soap_note.json",
        include_formatted: bool = True
    ):
        """
        Save SOAP note to file.
        
        Args:
            soap_dict: SOAP note dictionary
            output_path: Output file path
            include_formatted: Also save formatted text version
        """
        # Save JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(soap_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved JSON to: {output_path}")
        
        # Save formatted text version
        if include_formatted:
            text_path = output_path.replace('.json', '.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(self.format_soap_note(soap_dict))
            logger.info(f"✓ Saved formatted text to: {text_path}")


def main():
    """
    Main function demonstrating usage with error handling.
    """
    print("=" * 70)
    print("SOAP Note Generator - Production Version")
    print("=" * 70)
    print()
    
    try:
        # Initialize generator
        # Set use_4bit=True to reduce memory usage
        generator = SOAPNoteGenerator(
            model_name_or_path="stabilityai/stablelm-2-zephyr-1_6b",
            use_4bit=False  # Set to True if running out of memory
        )
        
        # Load transcript
        transcript_path = "data/conversation.txt"
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = f.read()
            logger.info(f"Loaded transcript from: {transcript_path}")
            logger.info(f"Transcript length: {len(transcript)} characters")
        except FileNotFoundError:
            logger.error(f"Transcript file not found: {transcript_path}")
            # Example transcript for testing
            transcript = """
            Doctor: Good morning! What brings you in today?
            Patient: I've been having severe headaches for the past week.
            Doctor: Can you describe the pain?
            Patient: It's a throbbing pain on the right side of my head.
            Doctor: Any other symptoms?
            Patient: Yes, I feel nauseous and sensitive to light.
            Doctor: Let me examine you. Your blood pressure is 120/80.
            Doctor: Based on your symptoms, this appears to be a migraine.
            Doctor: I'll prescribe some medication and we'll schedule a follow-up.
            """
            logger.info("Using example transcript for demonstration")
        
        print("\n" + "-" * 70)
        print("Generating SOAP note...")
        print("-" * 70 + "\n")
        
        # Generate SOAP note
        soap_note = generator.generate_soap_note(
            transcript,
            max_new_tokens=512,
            temperature=0.2,  # Lower = more consistent
            chunk_long_text=True
        )
        
        # Display results
        print("\n" + "=" * 70)
        print("JSON OUTPUT")
        print("=" * 70)
        print(json.dumps(soap_note, indent=2))
        
        print("\n" + "=" * 70)
        print("FORMATTED OUTPUT")
        print("=" * 70)
        print(generator.format_soap_note(soap_note))
        
        # Save to file
        generator.save_soap_note(
            soap_note,
            output_path="soap_note.json",
            include_formatted=True
        )
        
        print("\n" + "=" * 70)
        print("✓ SOAP note generation complete!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

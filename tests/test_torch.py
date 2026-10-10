import tempfile
import unittest
from pathlib import Path
import numpy as np
try:
    import torch
    import transformers
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Install inference extra for real PyTorch hook tests')
class TorchTests(unittest.TestCase):
    def test_prefill_only_correct_index_and_hook_cleanup(self):
        from jrs_rem.defense import Intervention
        layers=torch.nn.ModuleList([torch.nn.Identity(),torch.nn.Identity()])
        model=torch.nn.Module();model.model=torch.nn.Module()
        model.model.layers=layers;model.model.norm=torch.nn.Identity()
        delta=torch.tensor([[100.,100.],[1.,2.],[3.,4.]])
        with Intervention(model,delta,torch.tensor([True,True,True])):
            x=torch.zeros(1,3,2)
            for layer in layers:x=layer(x)
            x=model.model.norm(x)
            torch.testing.assert_close(x[0,-1],torch.tensor([4.,6.]))
            torch.testing.assert_close(x[0,0],torch.zeros(2))
            y=torch.zeros(1,1,2)
            for layer in layers:y=layer(y)
            y=model.model.norm(y)
            torch.testing.assert_close(y,torch.zeros_like(y))
        self.assertTrue(all(not layer._forward_hooks for layer in layers))

    def test_hooks_removed_on_exception(self):
        from jrs_rem.defense import Intervention
        layer=torch.nn.Identity();model=torch.nn.Module();model.model=torch.nn.Module()
        model.model.layers=torch.nn.ModuleList([layer]);model.model.norm=torch.nn.Identity()
        with self.assertRaises(RuntimeError):
            with Intervention(model,torch.ones(2,2),torch.ones(2,dtype=torch.bool)):
                raise RuntimeError('Synthetic generation failure')
        self.assertFalse(layer._forward_hooks)
        self.assertFalse(model.model.norm._forward_hooks)

    def test_signed_projection_and_independent_layer_gate(self):
        from jrs_rem.defense import correction
        from jrs_rem.directions import projection
        # Unit shift rows: below / exactly at / above threshold, opposite, zero.
        force=torch.tensor([[.19,(1-.19**2)**.5],[.2,(1-.2**2)**.5],
                            [.21,(1-.21**2)**.5],[-1.,0.],[0.,0.],[1.,-2.]])
        direction=torch.tensor([[1.,0.]]*5+[[1.,-1.]])
        vector,active=correction(torch.zeros_like(force),force,direction)
        np.testing.assert_array_equal(active.numpy(),[False,True,True,False,False,True])
        _,expected=projection(force.numpy(),direction.numpy())
        np.testing.assert_allclose(vector.numpy(),expected,atol=1e-6)
        self.assertLess(vector[-1,1].item(),0)
        _,zero_active=correction(torch.zeros(1,2),torch.zeros(1,2),torch.ones(1,2),0)
        self.assertFalse(zero_active.item())
        with self.assertRaises(ValueError):
            correction(torch.zeros_like(force),force,direction,mode='legacy_nonnegative')

    def test_tiny_real_llava_forward_and_generation(self):
        # Random 2-layer model, no pretrained weights or network. Checks real HF layout/cache.
        from transformers import LlavaConfig,LlavaForConditionalGeneration,LlamaConfig,CLIPVisionConfig
        from jrs_rem.llava import hidden_states, intervention_modules
        from jrs_rem.defense import Intervention
        text=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                         num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128,
                         pad_token_id=0,bos_token_id=1,eos_token_id=2)
        vision=CLIPVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                               num_attention_heads=2,image_size=8,patch_size=4)
        cfg=LlavaConfig(text_config=text.to_dict(),vision_config=vision.to_dict(),image_token_index=31,
                        vision_feature_layer=-2,image_seq_length=4)
        model=LlavaForConditionalGeneration(cfg).eval()
        inputs={'input_ids':torch.tensor([[1,31,31,31,31,4,5]]),
                'attention_mask':torch.ones(1,7,dtype=torch.long),'pixel_values':torch.randn(1,3,8,8)}
        hs=hidden_states(model,inputs)
        self.assertEqual(tuple(hs.shape),(3,16))
        # Each hook site must match the corresponding extracted feature, including final norm.
        observed={}
        def capture(index):
            def hook(module, args, output):
                h=output[0] if isinstance(output,tuple) else output
                observed[index]=h[0,-1].detach().clone()
            return hook
        handles=[m.register_forward_hook(capture(i+1)) for i,m in enumerate(intervention_modules(model))]
        reference=hidden_states(model,inputs)
        for handle in handles:handle.remove()
        for index,value in observed.items():torch.testing.assert_close(value,reference[index])
        # A nonzero final-layer correction must be applied after RMSNorm, exactly once.
        delta=torch.zeros_like(hs);delta[-1]=torch.linspace(-.3,.3,16)
        active=torch.tensor([False,False,True])
        with Intervention(model,delta,active):
            edited=hidden_states(model,inputs)
            second=hidden_states(model,inputs)
        torch.testing.assert_close(edited[:-1],hs[:-1])
        torch.testing.assert_close(edited[-1],hs[-1]+delta[-1])
        torch.testing.assert_close(second,hs)
        with torch.inference_mode():
            baseline=model.generate(**inputs,max_new_tokens=2,do_sample=False,use_cache=True)
            with Intervention(model,torch.zeros_like(hs),torch.ones(3,dtype=torch.bool)):
                output=model.generate(**inputs,max_new_tokens=2,do_sample=False,use_cache=True)
        torch.testing.assert_close(output,baseline)
        self.assertGreater(output.shape[-1],7)

if __name__=='__main__':unittest.main()

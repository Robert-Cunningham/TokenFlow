def replace_attention(module, ReplacementClass, **kwargs):
    attn_index = 0

    def inner_replace_attention(module, ReplacementClass, **kwargs):
        nonlocal attn_index

        for name, child in module.named_children():
            if hasattr(child, "base_attn"):
                setattr(module, name, ReplacementClass(child.base_attn, attn_index, **kwargs))
                attn_index += 1
                continue

            if str(type(child)) == "<class 'diffusers.models.attention_processor.Attention'>":
                setattr(module, name, ReplacementClass(child, attn_index, **kwargs))
                attn_index += 1
                continue

            inner_replace_attention(child, ReplacementClass, **kwargs)

    inner_replace_attention(module, ReplacementClass, **kwargs)


def collect_attention(module, collection_fn):
    attn_index = 0
    out = []

    def inner_collect_attention(module, collection_fn):
        nonlocal attn_index
        nonlocal out

        for name, child in module.named_children():
            if hasattr(child, "base_attn"):
                out.append(collection_fn(child))
                attn_index += 1
                continue

            # print("into", child.__class__)
            inner_collect_attention(child, collection_fn)
            # print("out of", child.__class__)

    inner_collect_attention(module, collection_fn)

    return out

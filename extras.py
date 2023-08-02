def gen_synth_video():
	out = []
	for t in range(10):
		current = np.zeros((768, 512, 3)).astype(np.uint8)
		sz = 20
		current[t*sz:(t+1)*sz, t*sz:(t+1)*sz, 0:3] = 255
		out.append(Image.fromarray(current))
	
	return out

def test():
	before = torch.tensor([
		[1, 1],
		[1, -1],
		[-1, 1],
		[-1, -1],
	], dtype=torch.float32)

	local = torch.tensor([
		[.1, -.1],
		[-.1, -.1],
		[-.1, .1],
		[-.1, -.1],
	], dtype=torch.float32)

	normed = local / torch.linalg.norm(local, dim=-1, keepdim=True) # [hw][c] (6144, 320) # TODO
	print(normed)
	before_attn_transitions = before.cuda() @ einops.rearrange(normed, "hw c -> c hw").cuda()
	print(before_attn_transitions)
	before_argmax = torch.argmax(before_attn_transitions, dim=0).cpu()

	print(before[before_argmax])
	return before_argmax

torch.where((all_attn_transitions[19][0]['before'] == torch.arange(0, len(all_attn_transitions[19][0]['before']))) == False)[0]
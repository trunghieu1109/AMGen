async def forward_183(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze the target molecule's substitution pattern to identify the required functional groups and their relative positions on the benzene ring. "
        "Critically assess the directing effects of each substituent and how they influence regioselectivity. "
        "Embed feedback by explicitly considering the impact of substituent activation/deactivation on subsequent steps to avoid mechanistic oversights."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Map each reagent and condition in the given options to their corresponding chemical transformations, "
        "including typical regioselectivity and compatibility in aromatic substitution sequences. "
        "Explicitly annotate repeated reagents with their mechanistic purpose (e.g., protection, deprotection, reinstallation). "
        "Incorporate feedback by preparing for a mechanistic feasibility check in the next subtask, avoiding assumptions about reagent redundancy or sequence order."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for mapping reagents and conditions."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Perform a detailed mechanistic feasibility and synthetic practicality evaluation of each reaction step in the provided sequences. "
        "For each intermediate, identify functional groups and verify compatibility with the next reagent, flagging any chemically impractical or unusual steps (e.g., nitration of free anilines, timing of sulfonation, feasibility of aryl ether formation via phenol alkylation). "
        "This subtask explicitly addresses previous failures by rejecting sequences with mechanistic flaws before further analysis."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Decide on the mechanistic feasibility and flag impractical sequences."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"]],
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Compare and refine the analysis of each sequence based on predicted intermediates, focusing on yield, regioselectivity, and overall synthetic practicality. "
        "Integrate mechanistic soundness and synthetic strategy considerations from prior subtasks to identify the most plausible high-yield synthetic route. "
        "This subtask incorporates reflexion on previous errors by emphasizing practical feasibility over theoretical substitution patterns alone."
    )
    critic_instruction4 = (
        "Please review and provide the limitations of provided solutions of synthetic sequences, highlighting practical feasibility and potential pitfalls."
    )
    cot_reflect_desc4 = {
        "instruction": cot_reflect_instruction4,
        "critic_instruction": critic_instruction4,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"], results3["thinking"], results3["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Generate a final recommendation of the optimal reaction sequence from the given options, supported by comprehensive mechanistic reasoning and synthesis strategy considerations. "
        "Ensure the recommendation explicitly addresses and overcomes the prior reasoning failures, providing clear justification for the chosen sequence's feasibility and efficiency."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Provide the final optimal synthetic sequence recommendation with detailed justification."
    )
    debate_desc5 = {
        "instruction": debate_instruction5,
        "final_decision_instruction": final_decision_instruction5,
        "input": [taskInfo, results4["thinking"], results4["answer"]],
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "temperature": 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5["thinking"], results5["answer"])
    return final_answer, logs

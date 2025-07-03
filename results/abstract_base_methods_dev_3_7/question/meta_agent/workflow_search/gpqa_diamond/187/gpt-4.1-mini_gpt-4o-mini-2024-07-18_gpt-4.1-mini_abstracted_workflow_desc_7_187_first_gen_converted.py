async def forward_187(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Subtask 1: Identify and clearly define the given lattice parameters of the rhombohedral crystal: interatomic distance (lattice parameter a = 10 Å) and lattice angles (α = β = γ = 30°). Output these parameters for use in subsequent calculations."
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_instruction=cot_instruction_1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context="user input"
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, identifying lattice parameters, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction_2 = "Subtask 2: Convert the given lattice parameters into the necessary form for interplanar distance calculation, including calculating the lattice constants and angles in radians if needed, and prepare the parameters for reciprocal lattice vector calculations."
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_instruction=cot_instruction_2,
        input_list=[taskInfo, results1['thinking'], results1['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user input", "thinking of subtask 1", "answer of subtask 1"]
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, converting lattice parameters, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_sc_instruction_3 = "Subtask 3: Calculate the volume of the rhombohedral unit cell using the lattice parameters (a and α) to facilitate the calculation of reciprocal lattice vectors."
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_instruction=cot_sc_instruction_3,
        input_list=[taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user input", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, calculating unit cell volume, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_sc_instruction_4 = "Subtask 4: Compute the reciprocal lattice parameters (a*, α*) from the direct lattice parameters, which are required to calculate the interplanar spacing for the (111) plane."
    results4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_sc_instruction=cot_sc_instruction_4,
        input_list=[taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user input", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results4['cot_agent'][idx].id}, computing reciprocal lattice parameters, thinking: {results4['list_thinking'][idx]}; answer: {results4['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_instruction_5 = "Subtask 5: Calculate the interplanar distance d_(111) for the (111) plane using the reciprocal lattice parameters and the Miller indices (h=1, k=1, l=1) with the appropriate formula for rhombohedral crystals."
    results5 = await self.cot(
        subtask_id="subtask_5",
        cot_instruction=cot_instruction_5,
        input_list=[taskInfo, results4['thinking'], results4['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user input", "thinking of subtask 4", "answer of subtask 4"]
    )
    agents.append(f"CoT agent {results5['cot_agent'].id}, calculating interplanar distance d_111, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    debate_instruction_6 = "Subtask 6: Compare the calculated interplanar distance with the given multiple-choice options and select the closest matching answer (A, B, C, or D)."
    final_decision_instruction_6 = "Subtask 6: Make final decision on the closest matching multiple-choice answer for the interplanar distance."

    debate_desc_6 = {
        "instruction": debate_instruction_6,
        "context": ["user query", results5['thinking'], results5['answer']],
        "input": [taskInfo, results5['thinking'], results5['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }

    final_decision_desc_6 = {
        "instruction": final_decision_instruction_6,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }

    results6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc=debate_desc_6,
        final_decision_desc=final_decision_desc_6,
        n_repeat=self.max_round
    )

    for round in range(self.max_round):
        for idx, agent in enumerate(results6['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, comparing and selecting closest answer, thinking: {results6['list_thinking'][round][idx].content}; answer: {results6['list_answer'][round][idx].content}")

    agents.append(f"Final Decision agent, selecting final answer, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'], sub_tasks, agents)
    return final_answer, logs

async def forward_191(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Subtask 1: Extract and define the geometric parameters and charge configuration: radius R of the conductor, radius r of the cavity, displacement s between centers, charge +q inside the cavity, and positions of point P with distances L (from conductor center) and l (from cavity center), including the angle θ between vectors l and s."
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_instruction=cot_instruction_1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query"],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"SC-CoT agent {results1['cot_agent'][idx].id}, extracting geometric parameters, thinking: {results1['list_thinking'][idx]}; answer: {results1['list_answer'][idx]}")
    thinking1 = results1['list_thinking'][0]
    answer1 = results1['list_answer'][0]
    sub_tasks.append(f"Subtask 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append(results1['subtask_desc'])

    cot_instruction_2 = "Subtask 2: Analyze the electrostatic properties of the uncharged spherical conductor with a cavity containing charge +q, focusing on the induced charges on the conductors inner and outer surfaces and the resulting external field characteristics."
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_instruction=cot_instruction_2,
        input_list=[taskInfo, thinking1, answer1],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 1", "answer of subtask 1"],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"SC-CoT agent {results2['cot_agent'][idx].id}, analyzing electrostatic properties, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    thinking2 = results2['list_thinking'][0]
    answer2 = results2['list_answer'][0]
    sub_tasks.append(f"Subtask 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append(results2['subtask_desc'])

    cot_instruction_3 = "Subtask 3: Classify the effect of the conductor on the external electric field, specifically that the conductor shields the external region so that the net external field is as if the conductor plus cavity system is neutral except for induced charges, and determine the effective position of the charge or image charges for field calculation."
    debate_desc_3 = {
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "input": [taskInfo, thinking2, answer2],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    results3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc_3,
        final_decision_desc={
            "instruction": "Subtask 3: Make final decision on the effective charge position and shielding effect.",
            "output": ["thinking", "answer"],
            "temperature": 0.0
        },
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results3['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, classifying conductor effect, thinking: {results3['list_thinking'][round][idx].content}; answer: {results3['list_answer'][round][idx].content}")
    thinking3 = results3['thinking']
    answer3 = results3['answer']
    sub_tasks.append(f"Subtask 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append(results3['subtask_desc'])

    cot_instruction_4 = "Subtask 4: Compute the magnitude of the electric field at point P outside the conductor, using the effective charge position and distances, incorporating the angle θ between vectors l and s, and applying Coulomb's law accordingly."
    results4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_sc_instruction=cot_instruction_4,
        input_list=[taskInfo, thinking3, answer3],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 3", "answer of subtask 3"],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"SC-CoT agent {results4['cot_agent'][idx].id}, computing electric field magnitude, thinking: {results4['list_thinking'][idx]}; answer: {results4['list_answer'][idx]}")
    thinking4 = results4['list_thinking'][0]
    answer4 = results4['list_answer'][0]
    sub_tasks.append(f"Subtask 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append(results4['subtask_desc'])

    debate_instruction_5 = "Subtask 5: Compare the computed electric field expression with the given multiple-choice options and select the correct choice (A, B, C, or D) that matches the derived formula."
    final_decision_instruction_5 = "Subtask 5: Make final decision on the correct multiple-choice answer."
    debate_desc_5 = {
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "input": [taskInfo, thinking4, answer4],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_5 = {
        "instruction": final_decision_instruction_5,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc_5,
        final_decision_desc=final_decision_desc_5,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results5['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, selecting correct choice, thinking: {results5['list_thinking'][round][idx].content}; answer: {results5['list_answer'][round][idx].content}")
    thinking5 = results5['thinking']
    answer5 = results5['answer']
    sub_tasks.append(f"Subtask 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs

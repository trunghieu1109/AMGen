async def forward_188(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Subtask 1: Define and extract the essential features of spontaneously-broken symmetry relevant to effective particles, including what it means for a particle to be associated with such a symmetry."
    results_1 = await self.cot(
        subtask_id="subtask_1",
        cot_instruction=cot_instruction_1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context="user input"
    )
    agents.append(f"CoT agent {results_1['cot_agent'].id}, defining spontaneously-broken symmetry, thinking: {results_1['thinking'].content}; answer: {results_1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results_1['thinking'].content}; answer - {results_1['answer'].content}")
    logs.append(results_1['subtask_desc'])

    cot_sc_instruction_2 = "Subtask 2: Extract and characterize the defining features of each effective particle (Magnon, Skyrmion, Pion, Phonon) focusing on their origin and relation to symmetry breaking."
    results_2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_instruction=cot_sc_instruction_2,
        input_list=[taskInfo, results_1['thinking'], results_1['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 1", "answer of subtask 1"],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results_2['cot_agent'][idx].id}, characterizing particles, thinking: {results_2['list_thinking'][idx]}; answer: {results_2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results_2['thinking'].content}; answer - {results_2['answer'].content}")
    logs.append(results_2['subtask_desc'])

    cot_sc_instruction_3 = "Subtask 3: Analyze and classify Magnon in terms of its association with spontaneously-broken symmetry based on its defining features."
    results_3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_instruction=cot_sc_instruction_3,
        input_list=[taskInfo, results_1['thinking'], results_1['answer'], results_2['thinking'], results_2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results_3['cot_agent'][idx].id}, analyzing Magnon, thinking: {results_3['list_thinking'][idx]}; answer: {results_3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results_3['thinking'].content}; answer - {results_3['answer'].content}")
    logs.append(results_3['subtask_desc'])

    cot_sc_instruction_4 = "Subtask 4: Analyze and classify Skyrmion in terms of its association with spontaneously-broken symmetry based on its defining features."
    results_4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_sc_instruction=cot_sc_instruction_4,
        input_list=[taskInfo, results_1['thinking'], results_1['answer'], results_2['thinking'], results_2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results_4['cot_agent'][idx].id}, analyzing Skyrmion, thinking: {results_4['list_thinking'][idx]}; answer: {results_4['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results_4['thinking'].content}; answer - {results_4['answer'].content}")
    logs.append(results_4['subtask_desc'])

    cot_sc_instruction_5 = "Subtask 5: Analyze and classify Pion in terms of its association with spontaneously-broken symmetry based on its defining features."
    results_5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_sc_instruction=cot_sc_instruction_5,
        input_list=[taskInfo, results_1['thinking'], results_1['answer'], results_2['thinking'], results_2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results_5['cot_agent'][idx].id}, analyzing Pion, thinking: {results_5['list_thinking'][idx]}; answer: {results_5['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results_5['thinking'].content}; answer - {results_5['answer'].content}")
    logs.append(results_5['subtask_desc'])

    cot_sc_instruction_6 = "Subtask 6: Analyze and classify Phonon in terms of its association with spontaneously-broken symmetry based on its defining features."
    results_6 = await self.sc_cot(
        subtask_id="subtask_6",
        cot_sc_instruction=cot_sc_instruction_6,
        input_list=[taskInfo, results_1['thinking'], results_1['answer'], results_2['thinking'], results_2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results_6['cot_agent'][idx].id}, analyzing Phonon, thinking: {results_6['list_thinking'][idx]}; answer: {results_6['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results_6['thinking'].content}; answer - {results_6['answer'].content}")
    logs.append(results_6['subtask_desc'])

    debate_instruction_7 = "Subtask 7: Evaluate all analyzed particles (Magnon, Skyrmion, Pion, Phonon) to identify which one is NOT associated with a spontaneously-broken symmetry and select the corresponding multiple-choice answer (A, B, C, or D)."
    final_decision_instruction_7 = "Subtask 7: Make final decision on which particle is not associated with spontaneously-broken symmetry."

    debate_desc_7 = {
        "instruction": debate_instruction_7,
        "context": ["user query", results_3['thinking'], results_3['answer'], results_4['thinking'], results_4['answer'], results_5['thinking'], results_5['answer'], results_6['thinking'], results_6['answer']],
        "input": [taskInfo, results_3['thinking'], results_3['answer'], results_4['thinking'], results_4['answer'], results_5['thinking'], results_5['answer'], results_6['thinking'], results_6['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }

    final_decision_desc_7 = {
        "instruction": final_decision_instruction_7,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }

    results_7 = await self.debate(
        subtask_id="subtask_7",
        debate_desc=debate_desc_7,
        final_decision_desc=final_decision_desc_7,
        n_repeat=self.max_round
    )

    for round in range(self.max_round):
        for idx, agent in enumerate(results_7['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, evaluating particles, thinking: {results_7['list_thinking'][round][idx].content}; answer: {results_7['list_answer'][round][idx].content}")

    agents.append(f"Final Decision agent, deciding particle not associated with spontaneously-broken symmetry, thinking: {results_7['thinking'].content}; answer: {results_7['answer'].content}")
    sub_tasks.append(f"Subtask 7 output: thinking - {results_7['thinking'].content}; answer - {results_7['answer'].content}")
    logs.append(results_7['subtask_desc'])

    final_answer = await self.make_final_answer(results_7['thinking'], results_7['answer'], sub_tasks, agents)
    return final_answer, logs

async def forward_188(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_reflect_instruction_1 = "Subtask 1: Define spontaneously-broken symmetry including internal and spatial symmetries (e.g., continuous translational and rotational symmetries). Discuss how phonons arise as Goldstone bosons from spontaneous breaking of continuous translational invariance in crystals." 
    critic_instruction_1 = "Please review the definition of spontaneously-broken symmetry and its inclusion of spatial symmetries and phonons as Goldstone bosons, and provide any limitations or missing aspects." 
    cot_reflect_desc_1 = {
        'instruction': cot_reflect_instruction_1,
        'input': [taskInfo],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query']
    }
    critic_desc_1 = {
        'instruction': critic_instruction_1,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    results_1 = await self.reflexion(
        subtask_id="subtask_1",
        cot_reflect_desc=cot_reflect_desc_1,
        critic_desc=critic_desc_1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results_1['cot_agent'].id}, defining spontaneously-broken symmetry, thinking: {results_1['list_thinking'][0].content}; answer: {results_1['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results_1['critic_agent'].id}, providing feedback, thinking: {results_1['list_feedback'][i].content}; answer: {results_1['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results_1['cot_agent'].id}, refining definition, thinking: {results_1['list_thinking'][i + 1].content}; answer: {results_1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results_1['thinking'].content}; answer - {results_1['answer'].content}")
    logs.append(results_1['subtask_desc'])

    cot_sc_instruction_2 = "Subtask 2: Characterize each effective particle (Magnon, Skyrmion, Pion, Phonon) focusing on their origin and relation to both internal and spatial symmetry breaking. Consider multiple perspectives to ensure accuracy." 
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

    cot_instruction_3 = "Subtask 3: Analyze and classify Magnon in terms of its association with spontaneously-broken symmetry, explicitly linking its defining features to internal symmetry breaking." 
    results_3 = await self.cot(
        subtask_id="subtask_3",
        cot_instruction=cot_instruction_3,
        input_list=[taskInfo, results_1['thinking'], results_1['answer'], results_2['thinking'], results_2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    )
    agents.append(f"CoT agent {results_3['cot_agent'].id}, analyzing Magnon, thinking: {results_3['thinking'].content}; answer: {results_3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results_3['thinking'].content}; answer - {results_3['answer'].content}")
    logs.append(results_3['subtask_desc'])

    cot_instruction_4 = "Subtask 4: Analyze and classify Skyrmion in terms of its association with spontaneously-broken symmetry, verifying its topological nature and relation to broken symmetry." 
    results_4 = await self.cot(
        subtask_id="subtask_4",
        cot_instruction=cot_instruction_4,
        input_list=[taskInfo, results_1['thinking'], results_1['answer'], results_2['thinking'], results_2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    )
    agents.append(f"CoT agent {results_4['cot_agent'].id}, analyzing Skyrmion, thinking: {results_4['thinking'].content}; answer: {results_4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results_4['thinking'].content}; answer - {results_4['answer'].content}")
    logs.append(results_4['subtask_desc'])

    cot_instruction_5 = "Subtask 5: Analyze and classify Pion in terms of its association with spontaneously-broken symmetry, explicitly mentioning Goldstone boson status and chiral symmetry breaking." 
    results_5 = await self.cot(
        subtask_id="subtask_5",
        cot_instruction=cot_instruction_5,
        input_list=[taskInfo, results_1['thinking'], results_1['answer'], results_2['thinking'], results_2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    )
    agents.append(f"CoT agent {results_5['cot_agent'].id}, analyzing Pion, thinking: {results_5['thinking'].content}; answer: {results_5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results_5['thinking'].content}; answer - {results_5['answer'].content}")
    logs.append(results_5['subtask_desc'])

    cot_reflect_instruction_6 = "Subtask 6: Analyze and classify Phonon in terms of its association with spontaneously-broken symmetry, verifying whether phonons can be understood as Goldstone bosons for broken continuous translational symmetry in crystals." 
    critic_instruction_6 = "Please review the classification of phonons and provide feedback on whether phonons arise from spontaneously-broken spatial symmetry as Goldstone bosons." 
    cot_reflect_desc_6 = {
        'instruction': cot_reflect_instruction_6,
        'input': [taskInfo, results_1['thinking'], results_1['answer'], results_2['thinking'], results_2['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1', 'thinking of subtask 2', 'answer of subtask 2']
    }
    critic_desc_6 = {
        'instruction': critic_instruction_6,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    results_6 = await self.reflexion(
        subtask_id="subtask_6",
        cot_reflect_desc=cot_reflect_desc_6,
        critic_desc=critic_desc_6,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results_6['cot_agent'].id}, analyzing Phonon, thinking: {results_6['list_thinking'][0].content}; answer: {results_6['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results_6['critic_agent'].id}, providing feedback, thinking: {results_6['list_feedback'][i].content}; answer: {results_6['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results_6['cot_agent'].id}, refining Phonon classification, thinking: {results_6['list_thinking'][i + 1].content}; answer: {results_6['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results_6['thinking'].content}; answer - {results_6['answer'].content}")
    logs.append(results_6['subtask_desc'])

    debate_instruction_7 = "Subtask 7: Evaluate all analyzed particles (Magnon, Skyrmion, Pion, Phonon) to identify which one is NOT associated with a spontaneously-broken symmetry. Have three independent reasoning chains propose the answer, then vote on the correct choice (A: Magnon, B: Skyrmion, C: Pion, D: Phonon). Agents should explicitly challenge and justify their choices to avoid groupthink and confirm correctness." 
    final_decision_instruction_7 = "Subtask 7: Make final decision on which particle is not associated with spontaneously-broken symmetry." 
    debate_desc_7 = {
        "instruction": debate_instruction_7,
        "context": [
            "user query",
            results_3['thinking'], results_3['answer'],
            results_4['thinking'], results_4['answer'],
            results_5['thinking'], results_5['answer'],
            results_6['thinking'], results_6['answer']
        ],
        "input": [
            taskInfo,
            results_3['thinking'], results_3['answer'],
            results_4['thinking'], results_4['answer'],
            results_5['thinking'], results_5['answer'],
            results_6['thinking'], results_6['answer']
        ],
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

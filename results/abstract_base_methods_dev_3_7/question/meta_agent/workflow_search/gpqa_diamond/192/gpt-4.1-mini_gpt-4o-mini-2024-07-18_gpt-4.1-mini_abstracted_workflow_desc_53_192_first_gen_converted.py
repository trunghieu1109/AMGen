async def forward_192(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction1 = "Subtask 1: Understand and express the relationship between parallax (plx) and distance (r), specifically that r = 1/plx, with context from the query."
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_instruction=cot_sc_instruction1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query"],
        n_repeat=self.max_sc
    )
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    for idx in range(self.max_sc):
        agents.append(f"SC_CoT agent {results1['cot_agent'][idx].id}, reasoning about parallax-distance relation, thinking: {results1['list_thinking'][idx]}; answer: {results1['list_answer'][idx]}")
    logs.append(results1['subtask_desc'])

    cot_reflect_instruction2 = "Subtask 2: Rewrite the given number of stars variation with parallax (1/plx^5) in terms of distance r using the relationship from Subtask 1."
    critic_instruction2 = "Please review the rewritten expression and provide feedback on its correctness and limitations."
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_desc2 = {
        'instruction': critic_instruction2,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results2 = await self.reflexion(
        subtask_id="subtask_2",
        cot_reflect_desc=cot_reflect_desc2,
        critic_desc=critic_desc2,
        n_repeat=self.max_round
    )
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, rewriting star count variation in terms of distance, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results2['critic_agent'].id}, feedback round {i}, thinking: {results2['list_feedback'][i].content}; answer: {results2['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining expression round {i}, thinking: {results2['list_thinking'][i + 1].content}; answer: {results2['list_answer'][i + 1].content}")
    
    cot_instruction3 = "Subtask 3: Determine how the number of stars varies with distance r per unit range of distance by simplifying the expression obtained in Subtask 2."
    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_instruction=cot_instruction3,
        input_list=[taskInfo, results2['thinking'], results2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", "thinking of subtask 2", "answer of subtask 2"]
    )
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    agents.append(f"CoT agent {results3['cot_agent'].id}, simplifying star count variation with distance, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")

    cot_reflect_instruction3 = "Subtask 3 Reflexion: Review the simplification and confirm the correctness of the variation of number of stars with distance r."
    critic_instruction3 = "Please provide feedback on the simplification and identify any errors or improvements."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    critic_desc3 = {
        'instruction': critic_instruction3,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results3_reflex = await self.reflexion(
        subtask_id="subtask_3_reflexion",
        cot_reflect_desc=cot_reflect_desc3,
        critic_desc=critic_desc3,
        n_repeat=self.max_round
    )
    sub_tasks.append(f"Subtask 3 Reflexion output: thinking - {results3_reflex['thinking'].content}; answer - {results3_reflex['answer'].content}")
    agents.append(f"Reflexion CoT agent {results3_reflex['cot_agent'].id}, reviewing simplification, thinking: {results3_reflex['list_thinking'][0].content}; answer: {results3_reflex['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results3_reflex['critic_agent'].id}, feedback round {i}, thinking: {results3_reflex['list_feedback'][i].content}; answer: {results3_reflex['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results3_reflex['cot_agent'].id}, refining simplification round {i}, thinking: {results3_reflex['list_thinking'][i + 1].content}; answer: {results3_reflex['list_answer'][i + 1].content}")

    debate_instruction4 = "Subtask 4: Match the derived variation of number of stars with distance r to the correct multiple-choice option (A, B, C, or D)."
    final_decision_instruction4 = "Subtask 4: Make final decision on the correct multiple-choice option representing the variation of number of stars with distance r."
    debate_desc4 = {
        "instruction": debate_instruction4,
        "context": ["user query", results3_reflex['thinking'], results3_reflex['answer']],
        "input": [taskInfo, results3_reflex['thinking'], results3_reflex['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc4 = {
        "instruction": final_decision_instruction4,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        final_decision_desc=final_decision_desc4,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, matching variation to choices, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding correct choice, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs

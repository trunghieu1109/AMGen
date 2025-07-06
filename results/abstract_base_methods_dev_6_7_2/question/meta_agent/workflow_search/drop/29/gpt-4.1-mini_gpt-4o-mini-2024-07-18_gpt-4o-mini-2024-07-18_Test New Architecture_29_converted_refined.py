async def forward_29(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_reflect_instruction1 = (
        "Sub-task 1: Calculate the number of people of German ancestry and those from US or Danish ancestry by explicitly multiplying the total population by each percentage, "
        "show full multiplication and round to the nearest integer for each count. Do not proceed until you verify these results are accurate. "
        "Then compute how many more Germans there were than the combined US and Danish ancestry counts."
    )
    critic_instruction1 = (
        "Please review the arithmetic calculations for German, US, and Danish ancestry counts, ensuring multiplication and rounding are correct and consistent with the original question. "
        "Provide feedback on any errors or inconsistencies."
    )
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query"]
    }
    critic_desc1 = {
        'instruction': critic_instruction1,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id="subtask_1",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=1
    )
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, calculating ancestry counts with explicit multiplication and rounding, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    agents.append(f"Critic agent {results1['critic_agent'].id}, feedback: {results1['list_feedback'][0].content}; correct: {results1['list_correct'][0].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['list_thinking'][0].content}; answer - {results1['list_answer'][0].content}")
    logs.append(results1['subtask_desc'])
    
    sc_cot_instruction2 = (
        "Sub-task 2: Independently generate two separate calculations for each ancestry count (German, US, Danish) based on the percentages and total population, "
        "then compare and resolve any discrepancies before finalizing the refined answer for how many more Germans there were."
    )
    N = self.max_sc
    sc_cot_desc2 = {
        'instruction': sc_cot_instruction2,
        'input': [taskInfo, results1['list_thinking'][0], results1['list_answer'][0]],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=sc_cot_desc2,
        n_repeat=N
    )
    for idx in range(N):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, independent calculation #{idx+1} for ancestry counts, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    debate_instruction3 = (
        "Sub-task 3: Debate between two agents on the best method to calculate how many more Germans there were than the combined US and Danish ancestry counts. "
        "Agent A uses direct subtraction of rounded counts; Agent B uses combined percentage difference then multiplication. "
        "Debate which method accurately reflects the data, then vote on the final answer."
    )
    final_decision_instruction3 = "Sub-task 3: Make final decision on the most accurate calculation method and final answer."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc3 = {
        'instruction': final_decision_instruction3,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        final_decision_desc=final_decision_desc3,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results3['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round+1}, arguing calculation methods, thinking: {results3['list_thinking'][round][idx].content}; answer: {results3['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding most accurate calculation, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    cot_reflect_instruction4 = (
        "Sub-task 4: Independently recompute the difference between German ancestry and combined US plus Danish ancestry counts, "
        "showing every step explicitly, then confirm or correct the final answer produced above."
    )
    critic_instruction4 = (
        "Please review the independent recomputation for arithmetic and logical consistency, and confirm if the final answer is correct or provide corrections."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    critic_desc4 = {
        'instruction': critic_instruction4,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results4 = await self.reflexion(
        subtask_id="subtask_4",
        cot_reflect_desc=cot_reflect_desc4,
        critic_desc=critic_desc4,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, independently recomputing final difference, thinking: {results4['list_thinking'][0].content}; answer: {results4['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results4['list_feedback']))):
        agents.append(f"Critic agent {results4['critic_agent'].id}, feedback: {results4['list_feedback'][i].content}; correct: {results4['list_correct'][i].content}")
        if i + 1 < len(results4['list_thinking']) and i + 1 < len(results4['list_answer']):
            agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining final answer, thinking: {results4['list_thinking'][i + 1].content}; answer: {results4['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs

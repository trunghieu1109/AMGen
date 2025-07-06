async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction1 = "Subtask 1: Extract the percentage of Native Americans in Orem from the 2000 census data provided in the passage."
    debate_desc1 = {
        "instruction": cot_instruction1,
        "context": ["user query"],
        "input": [taskInfo],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc1 = {
        "instruction": "Subtask 1: Make final decision on the extracted percentage of Native Americans in 2000.",
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc1,
        final_decision_desc=final_decision_desc1,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results1['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, extracting Native American percentage, thinking: {results1['list_thinking'][round][idx].content}; answer: {results1['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding Native American percentage, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    native_american_percentage = results1['answer'].content
    
    cot_sc_instruction2 = "Subtask 2: Calculate the percentage of people who were not Native American in 2000 by subtracting the Native American percentage from 100%."
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "input": [taskInfo, results1['thinking'], results1['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, calculating non-Native American percentage, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    cot_reflect_instruction3 = "Subtask 3: Format the final answer to clearly state the percentage of people who were not Native American in 2000, ensuring completeness and clarity as per the query's output format requirements."
    critic_instruction3 = "Please review the formatted final answer for completeness and clarity."
    cot_reflect_desc3 = {
        "instruction": cot_reflect_instruction3,
        "input": [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc3 = {
        "instruction": critic_instruction3,
        "output": ["feedback", "correct"],
        "temperature": 0.0
    }
    results3 = await self.reflexion(
        subtask_id="subtask_3",
        cot_reflect_desc=cot_reflect_desc3,
        critic_desc=critic_desc3,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, formatting final answer, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results3['critic_agent'].id}, providing feedback, thinking: {results3['list_feedback'][i].content}; answer: {results3['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining final answer, thinking: {results3['list_thinking'][i + 1].content}; answer: {results3['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'], sub_tasks, agents)
    return final_answer, logs

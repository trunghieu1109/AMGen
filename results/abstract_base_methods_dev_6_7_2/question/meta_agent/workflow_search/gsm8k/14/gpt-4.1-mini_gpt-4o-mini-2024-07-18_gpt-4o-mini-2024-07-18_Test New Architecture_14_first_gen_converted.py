async def forward_14(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction1 = "Subtask 1: Calculate the total daily energy demand for 2300 people (100% coverage)."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, calculating total daily energy demand for 2300 people, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    cot_instruction2 = "Subtask 2: Calculate the total daily energy provided by the current production of 4200 bottles, each covering 20% of one person's daily energy."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, calculating total energy from 4200 bottles, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    additional_bottles_needed = None
    for i in range(1):
        cot_instruction3 = "Subtask 3: Determine the deficit in energy coverage and calculate how many more bottles are needed to satisfy 100% of the daily energy needs of 2300 people."
        cot_agent_desc3 = {
            'instruction': cot_instruction3,
            'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
        }
        results3 = await self.cot(
            subtask_id="subtask_3",
            cot_agent_desc=cot_agent_desc3
        )
        agents.append(f"CoT agent {results3['cot_agent'].id}, calculating deficit and additional bottles needed, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
        sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
        logs.append(results3['subtask_desc'])
        additional_bottles_needed = results3['answer'].content
    
    cot_agent_instruction4 = "Subtask 4: Output the integer number of additional bottles required to meet the full daily energy needs of 2300 people."
    cot_agent_desc4 = {
        'instruction': cot_agent_instruction4,
        'input': [taskInfo, additional_bottles_needed],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results4 = await self.answer_generate(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4
    )
    agents.append(f"AnswerGenerate agent {results4['cot_agent'].id}, outputting final integer answer, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs
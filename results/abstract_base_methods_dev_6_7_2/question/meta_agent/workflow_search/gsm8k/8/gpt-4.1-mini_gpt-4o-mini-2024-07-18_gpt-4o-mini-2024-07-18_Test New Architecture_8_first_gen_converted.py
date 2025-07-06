async def forward_8(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start sequential
    # Control Flow 1: start loop for cost calculations of two package options
    package_options = [
        {'id': 1, 'package_size': 3, 'package_price': 2.50},
        {'id': 2, 'package_size': 2, 'package_price': 1.00}
    ]
    costs = {}
    for option in package_options:
        subtask_id = f"subtask_{option['id']}"
        cot_instruction = f"Sub-task {option['id']}: Calculate total cost for buying 18 flowers using packages of size {option['package_size']} at ${option['package_price']} each." 
        cot_agent_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results = await self.cot(
            subtask_id=subtask_id,
            cot_agent_desc=cot_agent_desc
        )
        agents.append(f"CoT agent {results['cot_agent'].id}, calculating total cost for package size {option['package_size']}, thinking: {results['thinking'].content}; answer: {results['answer'].content}")
        sub_tasks.append(f"Sub-task {option['id']} output: thinking - {results['thinking'].content}; answer - {results['answer'].content}")
        logs.append(results['subtask_desc'])
        costs[option['id']] = results
    # Control Flow 2: end loop
    
    # Subtask 4: Compare the total costs of the two package options to determine the better price using CoT
    cot_instruction4 = "Sub-task 4: Compare the total costs of buying 18 flowers using packages of 3 and packages of 2, determine which option is cheaper and by how much." 
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, costs[1]['thinking'], costs[1]['answer'], costs[2]['thinking'], costs[2]['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4
    )
    agents.append(f"CoT agent {results4['cot_agent'].id}, comparing total costs, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    # Subtask 5: Calculate the amount of money saved by buying 18 flowers at the better price using CoT
    cot_instruction5 = "Sub-task 5: Calculate the amount of money saved by buying 18 flowers at the better price option determined in Sub-task 4." 
    cot_agent_desc5 = {
        'instruction': cot_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5 = await self.cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_agent_desc5
    )
    agents.append(f"CoT agent {results5['cot_agent'].id}, calculating savings, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs

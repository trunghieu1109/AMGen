async def forward_77(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Identify and Classify Variables
    cot_instruction = "Sub-task 1: Identify the key variables and parameters involved in the problem, including the point charge q, its trajectory s(t), the field point r, the vector d, the velocity v, and the constants c, epsilon_o, and mu_o."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying key variables, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    # Stage 1: Classify Potentials
    cot_instruction = "Sub-task 2: Classify the types of potentials to be derived: scalar potential V and vector potential A."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, classifying potentials, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # Stage 2: Derive Scalar and Vector Potentials
    cot_instruction = "Sub-task 3: Derive the expression for the scalar potential V at the observation point r and time t, considering the earlier time tr and the distance d."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent([taskInfo, thinking2, answer2], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, deriving scalar potential, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    cot_instruction = "Sub-task 4: Derive the expression for the vector potential A at the observation point r and time t, using the derived scalar potential V and the velocity v."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent([taskInfo, thinking3, answer3], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, deriving vector potential, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    # Stage 3: Calculate Distances and Final Expressions
    cot_instruction = "Sub-task 5: Calculate the distance d from the point where the electromagnetic field was generated to the observation point r using the trajectory s(t) and the earlier time tr."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent([taskInfo, thinking1], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, calculating distance, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])
    
    cot_instruction = "Sub-task 6: Compute the final expressions for both the scalar potential V and vector potential A using the results from subtasks 3, 4, and 5."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent([taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing final expressions, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])
    
    # Stage 4: Evaluate and Select Correct Choice
    cot_instruction = "Sub-task 7: Evaluate the derived expressions for V and A against the provided choices to determine which one matches the derived results."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent([taskInfo, thinking6, answer6], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, evaluating derived expressions, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])
    
    cot_instruction = "Sub-task 8: Select the correct choice based on the evaluation in subtask 7."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent([taskInfo, thinking7, answer7], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, selecting correct choice, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Subtask 8 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer
async def forward_38(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    cot_instruction = "Sub-task 1: Analyze the wave function provided in the query, which is given as (a / sqrt(1 + x)) - 0.5*i, and identify the components of the wave function, including the variable a, the imaginary unit i, and the square root function. Understand the implications of the wave function in the context of quantum mechanics."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing wave function, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    cot_instruction_2 = "Sub-task 2: Classify the conditions under which the particles can exist, specifically noting the constraints that none of the particles are found at x<1 and x>3. This will help in understanding the valid range for the wave function and how it relates to the value of a."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, classifying conditions, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    cot_instruction_3 = "Sub-task 3: Transform the wave function into a usable form by substituting the valid range of x (1 < x < 3) into the wave function. This will allow for the evaluation of the wave function at specific points within the defined range."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, transforming wave function, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    cot_instruction_4 = "Sub-task 4: Evaluate the wave function at specific points (e.g., x=1 and x=3) to find the normalization condition. This will involve calculating the integral of the absolute square of the wave function over the range [1, 3] and setting it equal to 1 to solve for a."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating wave function, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    debate_instruction = "Sub-task 5: Select the correct numerical value of a from the provided choices (0.85, 1.1, 0.35, 0.6) based on the evaluation results from subtask 4."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) for role in ["Agent A", "Agent B"]]
    N_max = self.max_round
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], 
                                           debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking4, answer4] + all_thinking[r-1] + all_answer[r-1]
                thinking5, answer5 = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting value of a, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking[r].append(thinking5)
            all_answer[r].append(answer5)

    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking_final, answer_final = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], 
                                                 "Sub-task 5: Make final decision on the value of a.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final output, thinking: {thinking_final.content}; answer: {answer_final.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_final.content}; answer - {answer_final.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_final, answer_final, sub_tasks, agents)
    return final_answer
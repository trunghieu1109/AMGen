async def forward_53(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Identify the structure and necessary features for synthesis
    cot_instruction = "Sub-task 1: Identify the structure of 5-isopropyl-3,4-dimethylcyclohex-1-ene and determine the necessary features for its synthesis via ring-closing metathesis."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying structure and features, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    # Stage 2: Analyze starting materials and evaluate compatibility
    cot_instruction_2 = "Sub-task 2: Analyze the provided choices for starting materials to determine which one contains the necessary structural features to synthesize the target compound."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing starting materials, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # Sub-task 3: Evaluate structural compatibility
    cot_reflect_instruction = "Sub-task 3: Evaluate the structural compatibility of each choice with the target compound, focusing on the presence of double bonds and the correct carbon chain length."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_reflect_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, evaluating structural compatibility, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    # Sub-task 4: Select the most suitable starting material
    cot_instruction_4 = "Sub-task 4: Select the most suitable starting material based on the evaluation in subtask 3, ensuring it meets the criteria for successful ring-closing metathesis." 
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, selecting suitable starting material, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    # Final answer generation
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
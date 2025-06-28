async def forward_10(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Geometric Configuration and Properties
    
    # Sub-task 1: Identify and understand the geometric configuration
    cot_instruction_1 = "Sub-task 1: Identify and understand the geometric configuration of the rectangles and the circle, including the collinearity of points D, E, C, F and the cyclic nature of points A, D, H, G."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, understanding geometric configuration, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Use properties of cyclic quadrilaterals
    cot_instruction_2 = "Sub-task 2: Use the properties of cyclic quadrilaterals to establish relationships between the sides and angles of the quadrilateral ADHG."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, using cyclic quadrilateral properties, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Apply properties of collinearity
    cot_instruction_3 = "Sub-task 3: Apply the properties of collinearity to the points D, E, C, F to understand the linear relationship between these points."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, applying collinearity properties, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Calculate Length of Segment CE
    
    # Sub-task 4: Calculate the length of segment CE
    cot_instruction_4 = "Sub-task 4: Calculate the length of segment CE using the established relationships from the cyclic quadrilateral and collinear points."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating length of CE, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
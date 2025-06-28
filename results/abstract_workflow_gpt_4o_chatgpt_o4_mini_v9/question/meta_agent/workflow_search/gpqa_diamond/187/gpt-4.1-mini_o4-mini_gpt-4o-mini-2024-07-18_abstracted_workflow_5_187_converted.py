async def forward_187(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Extract and Define Given Parameters
    # Sub-task 1: Extract crystal parameters (rhombohedral system, a=10 Å, alpha=beta=gamma=30°)
    cot_instruction_0_1 = (
        "Sub-task 1: Extract and clearly define the given crystal parameters from the query: "
        "the crystal system (rhombohedral), interatomic distance (a = 10 Angstrom), and the lattice angles (alpha = beta = gamma = 30 degrees). "
        "This sets the foundational data for subsequent calculations.")
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, extracting crystal parameters, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    print("Step 0.1: ", sub_tasks[-1])

    # Sub-task 2: Identify Miller indices of the (111) plane and its significance in rhombohedral lattice
    cot_instruction_0_2 = (
        "Sub-task 2: Identify the Miller indices of the plane in question, which is the (111) plane, "
        "and understand its significance in the context of the rhombohedral crystal lattice.")
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, identifying Miller indices, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Analyze Lattice and Plane Orientation
    # Sub-task 3: Analyze rhombohedral lattice vectors and relationships
    cot_instruction_1_3 = (
        "Sub-task 3: Analyze the geometric and crystallographic properties of a rhombohedral lattice with given lattice parameters "
        "(a = 10 Angstrom, alpha = beta = gamma = 30 degrees) to determine the lattice vectors and their relationships.")
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await cot_agent_1_3([taskInfo, thinking_0_1, answer_0_1], cot_instruction_1_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_3.id}, analyzing lattice vectors, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    print("Step 1.3: ", sub_tasks[-1])

    # Sub-task 4: Classify (111) plane orientation relative to lattice vectors
    cot_instruction_1_4 = (
        "Sub-task 4: Classify the (111) plane in terms of its orientation relative to the lattice vectors "
        "and understand how to apply the Miller indices to the rhombohedral lattice system.")
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await cot_agent_1_4([taskInfo, thinking_0_2, answer_0_2, thinking_1_3, answer_1_3], cot_instruction_1_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_4.id}, classifying plane orientation, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    print("Step 1.4: ", sub_tasks[-1])

    # Stage 2: Derive and Calculate Interplanar Spacing
    # Sub-task 5: Derive formula for interplanar spacing d(hkl) in rhombohedral lattice
    cot_instruction_2_5 = (
        "Sub-task 5: Derive or recall the formula for the interplanar spacing (d-spacing) of the (hkl) plane in a rhombohedral lattice, "
        "incorporating the lattice parameters (a, alpha, beta, gamma) and Miller indices (h, k, l).")
    cot_agent_2_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_2_5, answer_2_5 = await cot_agent_2_5([taskInfo, thinking_1_3, answer_1_3, thinking_1_4, answer_1_4], cot_instruction_2_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_5.id}, deriving d-spacing formula, thinking: {thinking_2_5.content}; answer: {answer_2_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_2_5.content}; answer - {answer_2_5.content}")
    print("Step 2.5: ", sub_tasks[-1])

    # Sub-task 6: Calculate interplanar distance d(111) using formula and given parameters
    cot_instruction_2_6 = (
        "Sub-task 6: Calculate the interplanar distance d(111) using the derived formula and the given lattice parameters "
        "(a = 10 Angstrom, alpha = beta = gamma = 30 degrees) and Miller indices (1,1,1).")
    cot_agent_2_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_2_6, answer_2_6 = await cot_agent_2_6([taskInfo, thinking_2_5, answer_2_5], cot_instruction_2_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_6.id}, calculating d(111), thinking: {thinking_2_6.content}; answer: {answer_2_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_2_6.content}; answer - {answer_2_6.content}")
    print("Step 2.6: ", sub_tasks[-1])

    # Stage 3: Compare and Select Best Answer
    # Sub-task 7: Compare calculated d(111) with given multiple-choice options
    cot_instruction_3_7 = (
        "Sub-task 7: Compare the calculated interplanar distance d(111) with the provided multiple-choice options "
        "(9.54 Angstrom, 8.95 Angstrom, 9.08 Angstrom, 10.05 Angstrom) to identify the closest match.")
    cot_agent_3_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_3_7, answer_3_7 = await cot_agent_3_7([taskInfo, thinking_2_6, answer_2_6], cot_instruction_3_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_7.id}, comparing with choices, thinking: {thinking_3_7.content}; answer: {answer_3_7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking_3_7.content}; answer - {answer_3_7.content}")
    print("Step 3.7: ", sub_tasks[-1])

    # Sub-task 8: Select best answer choice based on comparison and physical plausibility
    cot_instruction_3_8 = (
        "Sub-task 8: Prioritize and select the best answer choice based on the comparison, "
        "ensuring the selected value is consistent with the calculation and physical plausibility.")
    cot_agent_3_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_3_8, answer_3_8 = await cot_agent_3_8([taskInfo, thinking_3_7, answer_3_7], cot_instruction_3_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_8.id}, selecting best answer, thinking: {thinking_3_8.content}; answer: {answer_3_8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking_3_8.content}; answer - {answer_3_8.content}")
    print("Step 3.8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_8, answer_3_8, sub_tasks, agents)
    return final_answer

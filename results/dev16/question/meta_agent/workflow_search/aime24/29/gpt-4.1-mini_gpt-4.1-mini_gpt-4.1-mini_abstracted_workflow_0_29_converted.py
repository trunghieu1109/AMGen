async def forward_29(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Stage 0: Problem Formalization and Structural Analysis
    
    # Sub-task 1: Formally define the problem setting and uniform color conditions (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Formally define the problem setting: specify the 5x5 grid, the two types of indistinguishable chips (white and black), "
        "and the placement rules including that each cell contains at most one chip. Explicitly state the uniform color conditions for rows and columns: "
        "if a row or column contains chips, all chips in that row or column must be of the same color. Avoid assumptions about maximality or emptiness at this stage.")
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formal problem definition, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step stage_0.subtask_1: ", sub_tasks[-1])
    
    # Sub-task 2: Precisely define maximality condition and assumptions about empty rows/columns (CoT)
    cot_instruction_0_2 = (
        "Sub-task 2: Precisely define the maximality condition: interpret maximality as the property that no additional chip can be added to any empty cell without violating the uniform color condition in its row or column. "
        "Explicitly clarify assumptions about empty rows and columns, including that empty rows or columns are considered uncolored and that maximality applies only to adding chips in empty cells. "
        "Emphasize that the empty grid configuration is not maximal and must be excluded.")
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, define maximality, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step stage_0.subtask_2: ", sub_tasks[-1])
    
    # Sub-task 3: Analyze structural implications of uniform color constraints (SC_CoT)
    cot_sc_instruction_0_3 = (
        "Sub-task 3: Analyze the structural implications of the uniform color constraints on rows and columns, focusing on the compatibility of color assignments at their intersections. "
        "Deduce necessary conditions for the color assignments of rows and columns to be consistent, such as that the intersection of a colored row and a colored column must have the same color chip. "
        "Highlight that this implies the grid can be partitioned into rectangular blocks of chips of uniform color corresponding to intersections of colored row and column subsets.")
    N_sc_0_3 = self.max_sc
    cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_3)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0_3):
        thinking_i, answer_i = await cot_agents_0_3[i]([taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, analyze uniform color structure, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_3.append(answer_i)
        possible_thinkings_0_3.append(thinking_i)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_thinkings_0_3, "Sub-task 3: Synthesize and choose the most consistent and correct structural analysis for uniform color constraints.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesize uniform color structure, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step stage_0.subtask_3: ", sub_tasks[-1])
    
    # Stage 1: Modeling and Characterization of Patterns
    
    # Sub-task 1: Model problem as assigning colors to rows and columns (CoT)
    cot_instruction_1_1 = (
        "Sub-task 1: Model the problem as assigning colors (white, black, or empty) to each row and column, ensuring compatibility at intersections. "
        "Define the possible color states for rows and columns and formalize the rules for their intersections to determine chip placements. "
        "Explicitly represent the combinatorial structure of these assignments, emphasizing that colored rows and columns form rectangular blocks of chips of the same color.")
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_3, answer_0_3], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, model color assignments, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step stage_1.subtask_1: ", sub_tasks[-1])
    
    # Sub-task 2: Translate maximality into combinatorial constraints on assignments (SC_CoT)
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Translate the maximality condition into explicit combinatorial constraints on the row and column color assignments. "
        "Determine when a row or column must be empty or colored, and how maximality restricts the possible extensions of chip placements. "
        "Emphasize that maximality requires that for each colored row (or column), all compatible columns (or rows) of the same color must be included to form a maximal rectangular block, forbidding partial intersections or incomplete blocks.")
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_0_2.content, answer_0_2.content, thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_0_2, answer_0_2, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, translate maximality constraints, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent maximality constraints on color assignments.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesize maximality constraints, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step stage_1.subtask_2: ", sub_tasks[-1])
    
    # Sub-task 3: Enumerate and characterize all possible compatible patterns (Reflexion)
    reflect_inst_1_3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_3 = (
        "Sub-task 3: Enumerate and characterize all possible compatible patterns of row and column color assignments that satisfy both the uniformity and maximality conditions, without yet counting their number. "
        "Explicitly exclude the trivial empty grid configuration. Identify the combinatorial dependencies between the sizes and intersections of colored row and column subsets for each color, ensuring that these subsets form complete rectangular blocks fully occupied by chips." + reflect_inst_1_3)
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_3 = self.max_round
    cot_inputs_1_3 = [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_reflect_instruction_1_3,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, characterize patterns, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    for i in range(N_max_1_3):
        feedback_1_3, correct_1_3 = await critic_agent_1_3([taskInfo, thinking_1_3], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, feedback: {feedback_1_3.content}; correct: {correct_1_3.content}")
        if correct_1_3.content == "True":
            break
        cot_inputs_1_3.extend([thinking_1_3, feedback_1_3])
        thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining patterns, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step stage_1.subtask_3: ", sub_tasks[-1])
    
    # Sub-task 4: Sanity check on characterized patterns (Reflexion)
    reflect_inst_1_4 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_4 = (
        "Sub-task 4: Perform a sanity check on the characterized patterns by testing extreme or boundary cases, such as all rows or columns colored the same, or minimal nonempty configurations. "
        "Verify that the maximality and uniformity conditions hold and that no invalid or trivial configurations are included. Use this to refine the characterization if necessary." + reflect_inst_1_4)
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_4 = self.max_round
    cot_inputs_1_4 = [taskInfo, thinking_1_3, answer_1_3]
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": cot_reflect_instruction_1_4,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_4, answer_1_4 = await cot_agent_1_4(cot_inputs_1_4, cot_reflect_instruction_1_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_4.id}, sanity check patterns, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    for i in range(N_max_1_4):
        feedback_1_4, correct_1_4 = await critic_agent_1_4([taskInfo, thinking_1_4], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_4.id}, feedback: {feedback_1_4.content}; correct: {correct_1_4.content}")
        if correct_1_4.content == "True":
            break
        cot_inputs_1_4.extend([thinking_1_4, feedback_1_4])
        thinking_1_4, answer_1_4 = await cot_agent_1_4(cot_inputs_1_4, cot_reflect_instruction_1_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_4.id}, refining sanity check, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step stage_1.subtask_4: ", sub_tasks[-1])
    
    # Stage 2: Counting Valid Maximal Placements
    
    # Sub-task 1: Develop detailed counting method incorporating combinatorial dependencies (CoT)
    cot_instruction_2_1 = (
        "Sub-task 1: Develop a detailed counting method to compute the number of valid maximal chip placements based on the characterized patterns of row and column color assignments. "
        "Break down the counting into enumerating all possible sizes of colored row and column subsets for each color, counting the number of ways to choose these subsets, and ensuring that the resulting rectangular blocks fit within the 5x5 grid and respect maximality constraints. "
        "Explicitly forbid the empty grid and partial blocks to avoid overcounting.")
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_3.content, thinking_1_4.content, answer_1_4.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_3, answer_1_3, thinking_1_4, answer_1_4], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, develop counting method, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step stage_2.subtask_1: ", sub_tasks[-1])
    
    # Sub-task 2: Apply counting method to 5x5 grid, sum over valid combinations (Reflexion)
    reflect_inst_2_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Apply the developed counting method to the specific 5x5 grid case, considering the indistinguishability of chips and the constraints on the number of chips available (25 white and 25 black). "
        "Sum over all valid combinations of colored row and column subsets to find the final number of valid maximal placements. Include explicit checks to prevent overcounting configurations where colored rows or columns lack matching colored columns or rows, respectively." + reflect_inst_2_2)
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1, thinking_1_4, answer_1_4]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content, thinking_1_4.content, answer_1_4.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, apply counting method, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback_2_2, correct_2_2 = await critic_agent_2_2([taskInfo, thinking_2_2], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, feedback: {feedback_2_2.content}; correct: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining counting application, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step stage_2.subtask_2: ", sub_tasks[-1])
    
    # Sub-task 3: Validate final counting result (Reflexion)
    reflect_inst_2_3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_3 = (
        "Sub-task 3: Validate the final counting result by cross-verifying that all problem constraints are respected, including maximality, uniformity, chip indistinguishability, and the exclusion of trivial or invalid configurations. "
        "If inconsistencies or overcounting are detected, identify and correct the counting steps accordingly." + reflect_inst_2_3)
    cot_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_3 = self.max_round
    cot_inputs_2_3 = [taskInfo, thinking_2_2, answer_2_2]
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_reflect_instruction_2_3,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_3, answer_2_3 = await cot_agent_2_3(cot_inputs_2_3, cot_reflect_instruction_2_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_3.id}, validate counting, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    for i in range(N_max_2_3):
        feedback_2_3, correct_2_3 = await critic_agent_2_3([taskInfo, thinking_2_3], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_3.id}, feedback: {feedback_2_3.content}; correct: {correct_2_3.content}")
        if correct_2_3.content == "True":
            break
        cot_inputs_2_3.extend([thinking_2_3, feedback_2_3])
        thinking_2_3, answer_2_3 = await cot_agent_2_3(cot_inputs_2_3, cot_reflect_instruction_2_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_3.id}, refining validation, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step stage_2.subtask_3: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_2_3, answer_2_3, sub_tasks, agents)
    return final_answer, logs

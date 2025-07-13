async def forward_13(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Derive rigorously from first principles the geometric relationship for a single circle tangent to both sides of the angle at vertex B of triangle ABC. "
        "Specifically, establish the formula for the distance d from vertex B to the center of a circle of radius r tangent to both sides, showing that d = r · csc(θ/2), where θ is the angle at vertex B. "
        "Avoid assuming any special triangle properties or unverified formulas. Provide a clear, step-by-step geometric proof or derivation."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_1: {subtask_desc1}")
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, deriving single circle tangent relation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, derive the exact formula governing a chain of n equal circles tangent sequentially to each other and tangent to both sides of the angle θ at vertex B. "
        "Derive the geometric progression ratio q = (1 - sin(θ/2)) / (1 + sin(θ/2)) that relates the distances between centers of consecutive circles. "
        "Formulate the closed-form expression for the total length of the chain in terms of r, n, and θ. Avoid any approximations or assumptions; the derivation must be explicit and mathematically rigorous."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before subtask_2: {subtask_desc2}")
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, deriving chain formula, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2 = "Sub-task 2: Synthesize and choose the most consistent and correct chain formula derivation based on all previous answers."
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, final_instr_2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Using the formulas derived in Sub-task 2, formulate the exact transcendental equations for the two given configurations: "
        "(a) 8 circles of radius 34, and (b) 2024 circles of radius 1, arranged sequentially tangent inside the angle θ. "
        "Set up the system of equations that must be satisfied simultaneously by θ and the inradius r_in of triangle ABC. "
        "Express these equations explicitly in terms of sin(θ/2), r, and n, avoiding any approximations."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_3: {subtask_desc3}")
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, formulating transcendental equations, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Numerically solve the transcendental system from Sub-task 3 with high precision. "
        "Implement a robust numerical method to find the accurate value of θ (angle at vertex B) and the inradius r_in = r · csc(θ/2). "
        "Avoid using small-angle approximations unless fully justified. Provide detailed steps, convergence criteria, and error bounds. "
        "Cross-validate the solution by substituting back into both circle configurations to ensure consistency and geometric feasibility."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before subtask_4: {subtask_desc4}")
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, numerically solving equations, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_4 = "Sub-task 4: Synthesize and choose the most accurate numerical solution for θ and inradius r_in based on all previous answers."
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_answers_4 + possible_thinkings_4, final_instr_4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 5: Express the computed inradius r_in as a reduced fraction m/n, where m and n are relatively prime positive integers. "
        "Perform exact fraction simplification, carefully checking for common factors. Provide a detailed explanation of the simplification process, including any rational approximations or continued fraction expansions used. "
        "Ensure the fraction accurately represents the numerical value of r_in within acceptable precision."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_5: {subtask_desc5}")
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, simplifying inradius fraction, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    debate_instr_6 = (
        "Sub-task 6: Compute the sum m + n from the simplified fraction representing the inradius. "
        "Verify the correctness of the final answer by cross-checking with the problem conditions, the derived geometric relationships, and the numerical solutions. "
        "Conduct a brief Reflexion or Debate stage where agents critique the final fraction and sum, ensuring no contradictions or errors remain. "
        "Provide a clear final answer along with a concise justification and verification summary."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instr_6,
        "context": ["user query", thinking5.content, answer5.content],
        "agent_collaboration": "Debate"
    }
    print(f"Logging before subtask_6: {subtask_desc6}")
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instr_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instr_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying final sum, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_6 = "Sub-task 6: Given all the above reasoning and verifications, provide the final sum m+n and a concise justification."
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], final_instr_6, is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs

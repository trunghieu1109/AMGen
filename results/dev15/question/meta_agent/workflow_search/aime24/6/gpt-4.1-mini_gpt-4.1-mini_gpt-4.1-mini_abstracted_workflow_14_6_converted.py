async def forward_6(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent_stage0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_stage0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_stage0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    cot_agent_stage1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_stage1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    N_sc = self.max_sc
    cot_sc_agents_stage1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    final_decision_agent_stage1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)

    cot_agent_stage2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_stage2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_stage2_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_stage2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)

    debate_agents_stage3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    final_decision_agent_stage3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)

    debate_agents_stage3_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    final_decision_agent_stage3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)

    # Stage 0 - Subtask 1
    cot_instruction_0_1 = "Sub-task 1: Identify and clearly state the domain of the problem: all triples (x, y, z) of positive real numbers representing the dimensions of rectangular boxes." 
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_stage0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_stage0_1.id}, identifying domain, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step stage_0.subtask_1: ", sub_tasks[-1])

    # Stage 0 - Subtask 2
    cot_instruction_0_2 = "Sub-task 2: Express the constraints defining the set B: surface area 2(xy + yz + zx) = 54 and volume xyz = 23." 
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_stage0_2([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_stage0_2.id}, expressing constraints, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step stage_0.subtask_2: ", sub_tasks[-1])

    # Stage 0 - Subtask 3
    cot_instruction_0_3 = "Sub-task 3: Explain the geometric interpretation of the minimal sphere containing a box: the sphere centered at the box's center with radius half the space diagonal r = sqrt(x^2 + y^2 + z^2)/2." 
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_stage0_3([taskInfo, thinking_0_1], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_stage0_3.id}, explaining geometry, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step stage_0.subtask_3: ", sub_tasks[-1])

    # Stage 1 - Subtask 1
    cot_instruction_1_1 = "Sub-task 1: Derive the expression for the squared space diagonal d^2 = x^2 + y^2 + z^2 in terms of the box dimensions." 
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_stage1_1([taskInfo, thinking_0_3], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_stage1_1.id}, deriving d^2, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step stage_1.subtask_1: ", sub_tasks[-1])

    # Stage 1 - Subtask 2
    cot_instruction_1_2 = "Sub-task 2: Rewrite the surface area constraint as xy + yz + zx = 27 to simplify the problem." 
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_stage1_2([taskInfo, thinking_0_2], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_stage1_2.id}, rewriting constraint, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step stage_1.subtask_2: ", sub_tasks[-1])

    # Stage 1 - Subtask 3 (SC-CoT)
    cot_sc_instruction_1_3 = "Sub-task 3: Formulate the constrained optimization problem: minimize d^2 = x^2 + y^2 + z^2 subject to xy + yz + zx = 27 and xyz = 23." 
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_1_1.content, thinking_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_stage1_3[i]([taskInfo, thinking_1_1, thinking_1_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_stage1_3[i].id}, formulating optimization, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_3.append(answer_i)
        possible_thinkings_1_3.append(thinking_i)
    thinking_1_3, answer_1_3 = await final_decision_agent_stage1_3([taskInfo] + possible_thinkings_1_3, "Sub-task 3: Synthesize and choose the most consistent formulation for the constrained optimization problem." + cot_sc_instruction_1_3, is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent_stage1_3.id}, synthesizing optimization formulation, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step stage_1.subtask_3: ", sub_tasks[-1])

    # Stage 2 - Subtask 1 (Reflexion)
    reflect_inst_2_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_1 = "Sub-task 1: Apply Lagrange multipliers or an equivalent method to find critical points of d^2 under the given constraints." + reflect_inst_2_1
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking_1_3.content],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_2_1 = [taskInfo, thinking_1_3]
    thinking_2_1, answer_2_1 = await cot_agent_stage2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_stage2_1.id}, applying Lagrange multipliers, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(self.max_round):
        feedback_2_1, correct_2_1 = await critic_agent_stage2([taskInfo, thinking_2_1], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_stage2.id}, feedback: {feedback_2_1.content}; correct: {correct_2_1.content}")
        if correct_2_1.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, feedback_2_1])
        thinking_2_1, answer_2_1 = await cot_agent_stage2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_stage2_1.id}, refining solution, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step stage_2.subtask_1: ", sub_tasks[-1])

    # Stage 2 - Subtask 2 (Reflexion)
    reflect_inst_2_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_2 = "Sub-task 2: Solve the resulting system of equations to find the values of x, y, z that minimize d^2." + reflect_inst_2_2
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_2_2 = [taskInfo, thinking_2_1]
    thinking_2_2, answer_2_2 = await cot_agent_stage2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_stage2_2.id}, solving system, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(self.max_round):
        feedback_2_2, correct_2_2 = await critic_agent_stage2([taskInfo, thinking_2_2], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_stage2.id}, feedback: {feedback_2_2.content}; correct: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_stage2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_stage2_2.id}, refining solution, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step stage_2.subtask_2: ", sub_tasks[-1])

    # Stage 2 - Subtask 3 (Reflexion)
    reflect_inst_2_3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_3 = "Sub-task 3: Verify that the found solution corresponds to a minimum and satisfies the positivity and constraint conditions." + reflect_inst_2_3
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_reflect_instruction_2_3,
        "context": ["user query", thinking_2_2.content],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_2_3 = [taskInfo, thinking_2_2]
    thinking_2_3, answer_2_3 = await cot_agent_stage2_3(cot_inputs_2_3, cot_reflect_instruction_2_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_stage2_3.id}, verifying solution, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    for i in range(self.max_round):
        feedback_2_3, correct_2_3 = await critic_agent_stage2([taskInfo, thinking_2_3], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_stage2.id}, feedback: {feedback_2_3.content}; correct: {correct_2_3.content}")
        if correct_2_3.content == "True":
            break
        cot_inputs_2_3.extend([thinking_2_3, feedback_2_3])
        thinking_2_3, answer_2_3 = await cot_agent_stage2_3(cot_inputs_2_3, cot_reflect_instruction_2_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_stage2_3.id}, refining verification, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step stage_2.subtask_3: ", sub_tasks[-1])

    # Stage 3 - Subtask 1 (Debate)
    debate_instr_3_1 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_3_1 = "Sub-task 1: Express the minimal squared radius r^2 = d^2/4 as a reduced fraction p/q with relatively prime positive integers p and q." + debate_instr_3_1
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking_2_3.content],
        "agent_collaboration": "Debate"
    }
    N_max_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(N_max_3_1)]
    all_answer_3_1 = [[] for _ in range(N_max_3_1)]
    for r in range(N_max_3_1):
        for i, agent in enumerate(debate_agents_stage3_1):
            if r == 0:
                thinking_3_1, answer_3_1 = await agent([taskInfo, thinking_2_3], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos_3_1 = [taskInfo, thinking_2_3] + all_thinking_3_1[r-1]
                thinking_3_1, answer_3_1 = await agent(input_infos_3_1, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, expressing minimal squared radius, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
            all_thinking_3_1[r].append(thinking_3_1)
            all_answer_3_1[r].append(answer_3_1)
    thinking_3_1, answer_3_1 = await final_decision_agent_stage3_1([taskInfo] + all_thinking_3_1[-1], "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_stage3_1.id}, finalizing minimal squared radius fraction, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task stage_3.subtask_1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step stage_3.subtask_1: ", sub_tasks[-1])

    # Stage 3 - Subtask 2 (Debate)
    debate_instr_3_2 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_3_2 = "Sub-task 2: Compute and report the sum p + q as the final answer." + debate_instr_3_2
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": debate_instruction_3_2,
        "context": ["user query", thinking_3_1.content],
        "agent_collaboration": "Debate"
    }
    N_max_3_2 = self.max_round
    all_thinking_3_2 = [[] for _ in range(N_max_3_2)]
    all_answer_3_2 = [[] for _ in range(N_max_3_2)]
    for r in range(N_max_3_2):
        for i, agent in enumerate(debate_agents_stage3_2):
            if r == 0:
                thinking_3_2, answer_3_2 = await agent([taskInfo, thinking_3_1], debate_instruction_3_2, r, is_sub_task=True)
            else:
                input_infos_3_2 = [taskInfo, thinking_3_1] + all_thinking_3_2[r-1]
                thinking_3_2, answer_3_2 = await agent(input_infos_3_2, debate_instruction_3_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing p+q, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
            all_thinking_3_2[r].append(thinking_3_2)
            all_answer_3_2[r].append(answer_3_2)
    thinking_3_2, answer_3_2 = await final_decision_agent_stage3_2([taskInfo] + all_thinking_3_2[-1], "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_stage3_2.id}, finalizing p+q sum, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    sub_tasks.append(f"Sub-task stage_3.subtask_2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step stage_3.subtask_2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs

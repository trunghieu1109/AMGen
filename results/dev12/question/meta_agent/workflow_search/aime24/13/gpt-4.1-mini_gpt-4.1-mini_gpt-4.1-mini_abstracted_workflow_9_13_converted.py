async def forward_13(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Formally model the geometric configuration of the chain of tangent circles inside triangle ABC. "
        "Define the arrangement of n equal circles tangent sequentially to each other and tangent to sides AB and BC at vertex B. "
        "Clarify the meaning of 'sequentially tangent' and establish the position of the chain along the angle at vertex B, avoiding assumptions about the inradius or other triangle elements at this stage."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, modeling geometric configuration, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    cot_instruction_2 = (
        "Sub-task 2: Derive from first principles the exact geometric relationship between the angle θ at vertex B, the number of tangent circles n, and their common radius r. "
        "Use coordinate geometry or distance-based reasoning to express a verified transcendental equation f(n, r, θ) = 0 that governs the chain configuration, ensuring no competing or unverified formulas are introduced."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, deriving geometric relation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)

    sc_cot_instruction_3 = (
        "Sub-task 3: Apply the derived relationship to the two given configurations: (n=8, r=34) and (n=2024, r=1). "
        "Formulate the system of equations relating θ, n, and r for both cases, preparing for numeric solving. Explicitly state the equations without attempting numeric solving here."
    )
    N_sc = self.max_sc
    sc_cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": sc_cot_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await sc_cot_agents_3[i]([taskInfo, thinking2, answer2], sc_cot_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_cot_agents_3[i].id}, formulating system of equations, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_answers_3 + possible_thinkings_3, "Sub-task 3: Synthesize and choose the most consistent system of equations for numeric solving.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    debate_instruction_4_1 = (
        "Sub-task 4.1: Numerically solve the transcendental equation(s) derived in stage_1.subtask_3 to find the angle θ at vertex B with high precision (residual error < 1E-6). "
        "Use numeric approximation or symbolic solvers, and document the numeric value of θ and the residuals to verify solution accuracy. Pass this numeric θ explicitly to subsequent subtasks. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_4_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4_1 = self.max_round
    all_thinking_4_1 = [[] for _ in range(N_max_4_1)]
    all_answer_4_1 = [[] for _ in range(N_max_4_1)]
    subtask_desc4_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_4_1,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4_1):
        for i, agent in enumerate(debate_agents_4_1):
            if r == 0:
                thinking4_1, answer4_1 = await agent([taskInfo, thinking3, answer3], debate_instruction_4_1, r, is_sub_task=True)
            else:
                input_infos_4_1 = [taskInfo, thinking3, answer3] + all_thinking_4_1[r-1] + all_answer_4_1[r-1]
                thinking4_1, answer4_1 = await agent(input_infos_4_1, debate_instruction_4_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, numeric solving θ, thinking: {thinking4_1.content}; answer: {answer4_1.content}")
            all_thinking_4_1[r].append(thinking4_1)
            all_answer_4_1[r].append(answer4_1)
    final_decision_agent_4_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4_1, answer4_1 = await final_decision_agent_4_1([taskInfo] + all_thinking_4_1[-1] + all_answer_4_1[-1], "Sub-task 4.1: Final numeric solution for θ with verification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4.1 output: thinking - {thinking4_1.content}; answer - {answer4_1.content}")
    subtask_desc4_1['response'] = {"thinking": thinking4_1, "answer": answer4_1}
    logs.append(subtask_desc4_1)

    sc_cot_instruction_4_2 = (
        "Sub-task 4.2: Using the numeric value of θ, independently compute the inradius of triangle ABC based on the verified geometric relations from stage_1.subtask_2. "
        "Avoid equating the inradius to any circle radius. Derive a numeric value for the inradius and prepare for fraction approximation."
    )
    N_sc_4_2 = self.max_sc
    sc_cot_agents_4_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_4_2)]
    possible_answers_4_2 = []
    possible_thinkings_4_2 = []
    subtask_desc4_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": sc_cot_instruction_4_2,
        "context": ["user query", thinking2.content, answer2.content, thinking4_1.content, answer4_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_4_2):
        thinking4_2, answer4_2 = await sc_cot_agents_4_2[i]([taskInfo, thinking2, answer2, thinking4_1, answer4_1], sc_cot_instruction_4_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_cot_agents_4_2[i].id}, computing inradius numerically, thinking: {thinking4_2.content}; answer: {answer4_2.content}")
        possible_answers_4_2.append(answer4_2)
        possible_thinkings_4_2.append(thinking4_2)
    final_decision_agent_4_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4_2, answer4_2 = await final_decision_agent_4_2([taskInfo] + possible_answers_4_2 + possible_thinkings_4_2, "Sub-task 4.2: Synthesize numeric inradius value.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4.2 output: thinking - {thinking4_2.content}; answer - {answer4_2.content}")
    subtask_desc4_2['response'] = {"thinking": thinking4_2, "answer": answer4_2}
    logs.append(subtask_desc4_2)

    reflect_instruction_5_1 = (
        "Sub-task 5.1: Convert the numeric inradius value into a fraction m/n in simplest terms, where m and n are relatively prime positive integers. "
        "Use numeric approximation techniques and gcd checks to ensure the fraction is fully reduced. Document the approximation error and justify the choice of fraction. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_5_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5_1 = self.max_round
    cot_inputs_5_1 = [taskInfo, thinking4_2, answer4_2]
    subtask_desc5_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": reflect_instruction_5_1,
        "context": ["user query", thinking4_2.content, answer4_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking5_1, answer5_1 = await cot_agent_5_1(cot_inputs_5_1, reflect_instruction_5_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5_1.id}, converting numeric inradius to fraction, thinking: {thinking5_1.content}; answer: {answer5_1.content}")
    for i in range(N_max_5_1):
        feedback5_1, correct5_1 = await critic_agent_5_1([taskInfo, thinking5_1, answer5_1], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5_1.id}, providing feedback, thinking: {feedback5_1.content}; answer: {correct5_1.content}")
        if correct5_1.content == "True":
            break
        cot_inputs_5_1.extend([thinking5_1, answer5_1, feedback5_1])
        thinking5_1, answer5_1 = await cot_agent_5_1(cot_inputs_5_1, reflect_instruction_5_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5_1.id}, refining fraction conversion, thinking: {thinking5_1.content}; answer: {answer5_1.content}")
    sub_tasks.append(f"Sub-task 5.1 output: thinking - {thinking5_1.content}; answer - {answer5_1.content}")
    subtask_desc5_1['response'] = {"thinking": thinking5_1, "answer": answer5_1}
    logs.append(subtask_desc5_1)

    debate_instruction_5_2 = (
        "Sub-task 5.2: Cross-validate the fraction m/n by comparing its decimal value to the numeric inradius computed earlier. "
        "Confirm the fraction's correctness within an acceptable tolerance and re-check gcd to ensure simplest form. "
        "If multiple candidate fractions arise, provide numeric evidence to select the correct one. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_5_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5_2 = self.max_round
    all_thinking_5_2 = [[] for _ in range(N_max_5_2)]
    all_answer_5_2 = [[] for _ in range(N_max_5_2)]
    subtask_desc5_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": debate_instruction_5_2,
        "context": ["user query", thinking5_1.content, answer5_1.content, thinking4_2.content, answer4_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5_2):
        for i, agent in enumerate(debate_agents_5_2):
            if r == 0:
                thinking5_2, answer5_2 = await agent([taskInfo, thinking5_1, answer5_1, thinking4_2, answer4_2], debate_instruction_5_2, r, is_sub_task=True)
            else:
                input_infos_5_2 = [taskInfo, thinking5_1, answer5_1, thinking4_2, answer4_2] + all_thinking_5_2[r-1] + all_answer_5_2[r-1]
                thinking5_2, answer5_2 = await agent(input_infos_5_2, debate_instruction_5_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, validating fraction, thinking: {thinking5_2.content}; answer: {answer5_2.content}")
            all_thinking_5_2[r].append(thinking5_2)
            all_answer_5_2[r].append(answer5_2)
    final_decision_agent_5_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5_2, answer5_2 = await final_decision_agent_5_2([taskInfo] + all_thinking_5_2[-1] + all_answer_5_2[-1], "Sub-task 5.2: Final fraction validation and selection.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5.2 output: thinking - {thinking5_2.content}; answer - {answer5_2.content}")
    subtask_desc5_2['response'] = {"thinking": thinking5_2, "answer": answer5_2}
    logs.append(subtask_desc5_2)

    sc_cot_instruction_6 = (
        "Sub-task 6: Compute the sum m + n from the simplified fraction representing the inradius and present it as the final answer. "
        "Ensure this step only proceeds after fraction correctness is confirmed."
    )
    sc_cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "stage_4.subtask_1",
        "instruction": sc_cot_instruction_6,
        "context": ["user query", thinking5_2.content, answer5_2.content],
        "agent_collaboration": "SC_CoT"
    }
    thinking6, answer6 = await sc_cot_agent_6([taskInfo, thinking5_2, answer5_2], sc_cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {sc_cot_agent_6.id}, computing final sum m+n, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
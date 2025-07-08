async def forward_192(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Understand and state the mathematical relationship between parallax (plx) and distance (r), specifically that plx = 1/r, and clarify the implications of this inverse proportionality."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, understanding parallax-distance relationship, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    cot_sc_instruction_2a = "Sub-task 2a: Determine whether the given relation number of stars varies with parallax as 1/plx^5 refers to the cumulative number of stars up to a parallax value or the differential number of stars per unit parallax interval (dN/dplx). Explicitly identify if the relation represents total counts N(<plx) or differential counts dN/dplx."
    N = self.max_sc
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2a, answer2a = await cot_agents_2a[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, determining cumulative vs differential star count relation, thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a.content)
        thinkingmapping_2a[answer2a.content] = thinking2a
        answermapping_2a[answer2a.content] = answer2a
    answer2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking2a = thinkingmapping_2a[answer2a_content]
    answer2a = answermapping_2a[answer2a_content]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {
        "thinking": thinking2a,
        "answer": answer2a
    }
    logs.append(subtask_desc2a)
    if "differential" in answer2a_content.lower():
        cot_instruction_2b = "Sub-task 2b: Assuming the relation is differential (dN/dplx ~ 1/plx^5), express the differential number of stars per unit parallax interval in terms of distance r, using the relationship plx = 1/r and applying the appropriate Jacobian transformation (dN/dplx to dN/dr)."
        cot_agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
        subtask_desc2b = {
            "subtask_id": "subtask_2b",
            "instruction": cot_instruction_2b,
            "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2a", "answer of subtask 2a"],
            "agent_collaboration": "CoT"
        }
        thinking2b, answer2b = await cot_agent_2b([taskInfo, thinking1, answer1, thinking2a, answer2a], cot_instruction_2b, is_sub_task=True)
        agents.append(f"CoT agent {cot_agent_2b.id}, expressing differential star count in terms of distance r, thinking: {thinking2b.content}; answer: {answer2b.content}")
        sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
        subtask_desc2b['response'] = {
            "thinking": thinking2b,
            "answer": answer2b
        }
        logs.append(subtask_desc2b)
        thinking2c = None
        answer2c = None
    else:
        cot_instruction_2c = "Sub-task 2c: Assuming the relation is cumulative (N(<plx) ~ 1/plx^5), convert the cumulative count to a differential count per unit parallax interval (dN/dplx), then express dN/dplx in terms of distance r using plx = 1/r and apply the Jacobian transformation to get dN/dr."
        cot_agent_2c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
        subtask_desc2c = {
            "subtask_id": "subtask_2c",
            "instruction": cot_instruction_2c,
            "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2a", "answer of subtask 2a"],
            "agent_collaboration": "CoT"
        }
        thinking2c, answer2c = await cot_agent_2c([taskInfo, thinking1, answer1, thinking2a, answer2a], cot_instruction_2c, is_sub_task=True)
        agents.append(f"CoT agent {cot_agent_2c.id}, converting cumulative to differential star count and expressing in terms of distance r, thinking: {thinking2c.content}; answer: {answer2c.content}")
        sub_tasks.append(f"Sub-task 2c output: thinking - {thinking2c.content}; answer - {answer2c.content}")
        subtask_desc2c['response'] = {
            "thinking": thinking2c,
            "answer": answer2c
        }
        logs.append(subtask_desc2c)
        thinking2b = thinking2c
        answer2b = answer2c
    cot_reflect_instruction_2d = "Sub-task 2d: Conduct a self-consistency check and reflexion step to confirm the correct interpretation (cumulative vs differential) and validate the transformation from parallax to distance, ensuring no misinterpretation propagates to subsequent steps."
    cot_agent_2d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2d = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2d = self.max_round
    cot_inputs_2d = [taskInfo, thinking1, answer1, thinking2a, answer2a]
    if thinking2b is not None and answer2b is not None:
        cot_inputs_2d.extend([thinking2b, answer2b])
    subtask_desc2d = {
        "subtask_id": "subtask_2d",
        "instruction": cot_reflect_instruction_2d,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2a", "answer of subtask 2a", "thinking of subtask 2b/2c", "answer of subtask 2b/2c"],
        "agent_collaboration": "Reflexion"
    }
    thinking2d, answer2d = await cot_agent_2d(cot_inputs_2d, cot_reflect_instruction_2d, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2d.id}, confirming interpretation and transformation, thinking: {thinking2d.content}; answer: {answer2d.content}")
    for i in range(N_max_2d):
        feedback, correct = await critic_agent_2d([taskInfo, thinking2d, answer2d], "please review the interpretation and transformation from parallax to distance and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2d.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2d.extend([thinking2d, answer2d, feedback])
        thinking2d, answer2d = await cot_agent_2d(cot_inputs_2d, cot_reflect_instruction_2d, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2d.id}, refining interpretation and transformation, thinking: {thinking2d.content}; answer: {answer2d.content}")
    sub_tasks.append(f"Sub-task 2d output: thinking - {thinking2d.content}; answer - {answer2d.content}")
    subtask_desc2d['response'] = {
        "thinking": thinking2d,
        "answer": answer2d
    }
    logs.append(subtask_desc2d)
    cot_instruction_3 = "Sub-task 3: Derive the explicit functional form of the number of stars per unit distance interval (dN/dr) in terms of r, based on the confirmed interpretation and transformations from stage 1."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2d, answer2d]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2d", "answer of subtask 2d"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, deriving functional form of dN/dr, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the derivation of the number of stars per unit distance interval and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining functional form of dN/dr, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    debate_instruction_4 = "Sub-task 4: Compare the derived expression for the number of stars per unit distance interval (dN/dr) with the given multiple-choice options (~ r^2, ~ r^4, ~ r^3, ~ r^5) and select the correct choice (A, B, C, or D)."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing derived expression with choices, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the correct choice for star count variation with distance r.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct choice, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs

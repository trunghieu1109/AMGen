async def forward_191(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Analyze the physical setup of an uncharged spherical conductor of radius R containing a spherical cavity of radius r (r < R), with the cavity center displaced by distance s from the conductor's center. Identify all relevant geometric parameters including distances L (from conductor center to point P), l (from cavity center to point P), and the angle theta between vectors l and s, with context from the query."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing physical setup, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Determine the electrostatic conditions inside and outside the conductor when a positive charge +q is placed inside the cavity, emphasizing that the conductor is isolated and initially uncharged, and that electrostatic shielding applies inside the conductor material."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining electrostatic conditions, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_reflect_instruction_3 = "Sub-task 3: Determine the induced charge distribution on the conductor surfaces due to the charge +q inside the cavity, explicitly distinguishing between induced charges on the inner cavity surface and the outer surface of the conductor, and confirm that the net charge on the conductor remains zero."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, determining induced charge distribution, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the induced charge distribution on inner and outer surfaces and confirm net conductor charge is zero.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining induced charge distribution, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_instruction_4a = "Sub-task 4a: Calculate the net induced charge on the outer surface of the conductor, which equals +q, and explain how this induced charge distribution affects the external electric field."
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_instruction_4a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking4a, answer4a = await cot_agent_4a([taskInfo, thinking3, answer3], cot_instruction_4a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4a.id}, calculating induced charge on outer surface, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {
        "thinking": thinking4a,
        "answer": answer4a
    }
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    
    cot_instruction_4b = "Sub-task 4b: Using the induced charge +q on the conductor's outer surface, compute the magnitude of the electric field at point P (distance L > R from the conductor center), treating the conductor as if it carries a net charge +q located at its center for points outside the conductor."
    cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "CoT"
    }
    thinking4b, answer4b = await cot_agent_4b([taskInfo, thinking4a, answer4a], cot_instruction_4b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4b.id}, computing electric field magnitude at P, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {
        "thinking": thinking4b,
        "answer": answer4b
    }
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    
    cot_reflect_instruction_4_reflexion = "Sub-task 4 Reflexion: Compare the results of Sub-tasks 2, 3, 4a, and 4b to ensure consistency in the induced charge distribution and the computed external electric field magnitude. If inconsistencies arise, re-evaluate the induced charge calculation and electric field computation."
    cot_agent_4r = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4r = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4r = self.max_round
    cot_inputs_4r = [taskInfo, thinking2, answer2, thinking3, answer3, thinking4a, answer4a, thinking4b, answer4b]
    subtask_desc4r = {
        "subtask_id": "subtask_4_reflexion",
        "instruction": cot_reflect_instruction_4_reflexion,
        "context": ["user query", "thinking and answer of subtask 2", "thinking and answer of subtask 3", "thinking and answer of subtask 4a", "thinking and answer of subtask 4b"],
        "agent_collaboration": "Reflexion"
    }
    thinking4r, answer4r = await cot_agent_4r(cot_inputs_4r, cot_reflect_instruction_4_reflexion, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4r.id}, comparing induced charge and electric field results, thinking: {thinking4r.content}; answer: {answer4r.content}")
    for i in range(N_max_4r):
        feedback, correct = await critic_agent_4r([taskInfo, thinking4r, answer4r], "please verify consistency of induced charge and electric field results and suggest corrections if needed.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4r.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4r.extend([thinking4r, answer4r, feedback])
        thinking4r, answer4r = await cot_agent_4r(cot_inputs_4r, cot_reflect_instruction_4_reflexion, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4r.id}, refining consistency check, thinking: {thinking4r.content}; answer: {answer4r.content}")
    sub_tasks.append(f"Sub-task 4 Reflexion output: thinking - {thinking4r.content}; answer - {answer4r.content}")
    subtask_desc4r['response'] = {
        "thinking": thinking4r,
        "answer": answer4r
    }
    logs.append(subtask_desc4r)
    print("Step 4 Reflexion: ", sub_tasks[-1])
    
    cot_sc_instruction_5 = "Sub-task 5: Compare the computed electric field magnitude at point P with the given multiple-choice options (A, B, C, or D), and select the correct choice based on the physical reasoning that the external field is due solely to the induced charge +q on the conductor's outer surface, effectively located at the conductor's center. Ensure the final answer is exactly one letter from A to D."
    N = self.max_sc
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_5 = []
    thinkingmapping_5 = {}
    answermapping_5 = {}
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", "thinking of subtask 4 Reflexion", "answer of subtask 4 Reflexion"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking4r, answer4r], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, selecting correct multiple-choice option, thinking: {thinking5.content}; answer: {answer5.content}")
        ans = answer5.content.strip().upper()
        if ans in ['A', 'B', 'C', 'D']:
            possible_answers_5.append(ans)
            thinkingmapping_5[ans] = thinking5
            answermapping_5[ans] = answer5
    if possible_answers_5:
        answer5_content = Counter(possible_answers_5).most_common(1)[0][0]
        thinking5 = thinkingmapping_5[answer5_content]
        answer5 = answermapping_5[answer5_content]
    else:
        answer5_content = 'D'
        thinking5 = None
        answer5 = None
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content if thinking5 else 'N/A'}; answer - {answer5_content}")
    subtask_desc5['response'] = {
        "thinking": thinking5 if thinking5 else "No valid thinking output",
        "answer": answer5_content
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking5, answer5_content, sub_tasks, agents)
    return final_answer, logs